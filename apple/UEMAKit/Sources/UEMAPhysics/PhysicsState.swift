// PhysicsState.swift: the value types and the three small protocols of the UEMAPhysics
// module beside MuJoCoPhysics (work item WI-25, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; the other file of the item is
// MuJoCoPhysics.swift). Nothing in this file names a MuJoCo C symbol, so a fake of any
// protocol here compiles in a test target without the MuJoCo XCFramework.
//
// WHAT IS HERE, ONE RESPONSIBILITY EACH.
//   PhysicsError             the errors MuJoCoPhysics throws, each naming the measured value.
//   PhysicsState             a copy of the state arrays at one instant (time, qpos, qvel, ctrl,
//                            sensordata, xpos, xquat, actuator_force, contact count) with the
//                            finiteness check of the work item's acceptance ("no NaN in qpos").
//   PhysicsModelTables       the model constants a reader or the command decoder needs (names,
//                            joint and actuator tables, qpos0, timestep), read once from mjModel.
//   PhysicsModelDescribing   gives a PhysicsModelTables: the "model-description protocol (WI-25)"
//                            that UEMACore's PhysicsReading doc comment names
//                            (Sources/UEMACore/Protocols.swift:380-383).
//   ActuatorParameterWriting the two model writes of the body-function modulation:
//                            actuator_gainprm[:, 0] and actuator_forcerange (env.py:2832-2845).
//   ScratchKinematicsQuerying a body's world position with chosen joint angles, computed on a
//                            scratch mjData (action_decoder.py:224-238), for the reach
//                            measurement of the command decoder (WI-37).
//
// WHY THE LAST THREE LIVE HERE. UEMACommand (WI-37) depends on UEMAPhysics, never the reverse
// (Package.swift, module sequence). Its CommandDecoder.swift declares BodyStateReading (line 572)
// and ForwardKinematicsQuerying (line 591) and adapts them over the protocols below in its
// "Adapters over UEMAPhysics (WI-25)" section (CommandDecoder.swift:1541-1612):
//   ActuatedBodyTables(modelTables:)   <- PhysicsModelTables fields of the same mjModel names
//   ScratchForwardKinematics           <- ScratchKinematicsQuerying.scratchBodyPosition (3 numbers
//                                         to Vector3)
// The decoder writes data.ctrl through UEMACore's Physics.write(controls: MotorControls), which
// throws on the released scene's 90 motors (MuJoCoPhysics.swift header, THE MOTOR COUNT);
// MuJoCoPhysics.write(controls: [Double]) takes nu numbers and runs today.
// The tactile, proprioception and vestibular readers of UEMASenses (WI-27, WI-28) also need
// geom_bodyid (ngeom) and body_rootid (nbody); PhysicsModelTables carries both
// (ProprioceptionReader.swift:299 and PhysicsContactView.swift:44 take them as arguments).
//
// Nothing here was compiled on the machine that wrote it (Windows 11, no Swift toolchain; plan
// section 3, fact 1).
//
// FIRST BUILD ON A MAC: see the checklist in MuJoCoPhysics.swift's header; this file has no
// build step of its own.

import Foundation
import UEMACore

// MARK: - Errors

/// The errors of the UEMAPhysics module. Each case carries the values that let a reader judge
/// the failure without rerunning it.
public enum PhysicsError: Error, Equatable, Sendable {
    /// A read or step was asked of a physics with no scene loaded.
    case notLoaded
    /// The scene file was not found at `path`.
    case sceneFileMissing(path: String)
    /// mj_loadXML returned NULL; `message` is MuJoCo's error text from its error buffer.
    case loadFailed(path: String, message: String)
    /// mj_makeData returned NULL for the loaded model.
    case dataAllocationFailed
    /// The scene's physics timestep (mjModel.opt.timestep, seconds) is not the 0.005 s a model
    /// step is built from (ModelStep.physicsTimestepSeconds; mimo_env.py frame_skip 2 at env.py:1157).
    case timestepMismatch(expectedSeconds: Double, foundSeconds: Double)
    /// An array of `name` had `found` numbers where the loaded model needs `expected`.
    case wrongCount(name: String, expected: Int, found: Int)
    /// Number `index` of `name` was NaN or infinite.
    case nonFinite(name: String, index: Int)
    /// An object id lies outside 0 ..< `count` of `kind`.
    case idOutOfRange(kind: MuJoCoObjectKind, id: Int, count: Int)
    /// A contact id lies outside 0 ..< `count` (mjData.ncon of the current physics step).
    case contactOutOfRange(id: Int, count: Int)
    /// The model has no object of `kind` named `name`.
    case unknownName(kind: MuJoCoObjectKind, name: String)
    /// mjv_makeScene left a geom buffer of 0 (it allocates `requested` geoms on success).
    case sceneAllocationFailed(requested: Int)
}

// MARK: - The state at one instant

/// A copy of the physics state at one instant, taken through `PhysicsReading`. Units are
/// MuJoCo's: seconds, metres, radians, newtons. Arrays are flat and row-major (`xpos` nbody x 3,
/// `xquat` nbody x 4 in w x y z sequence). `contactCount` is mjData.ncon.
public struct PhysicsState: Equatable, Codable, Sendable {
    /// Simulated time in seconds (mjData.time).
    public let time: Double
    /// Generalized positions, nq (146 on the released crib scene, bundle.json model_counts.nq).
    public let qpos: [Double]
    /// Generalized velocities, nv (136 on the released crib scene).
    public let qvel: [Double]
    /// The controls in mjData.ctrl, nu.
    public let control: [Double]
    /// The sensor values, nsensordata.
    public let sensordata: [Double]
    /// Body positions, nbody x 3, metres.
    public let xpos: [Double]
    /// Body orientations, nbody x 4, quaternion w x y z.
    public let xquat: [Double]
    /// Actuator forces in actuation space, nu.
    public let actuatorForce: [Double]
    /// The number of contacts of the current physics step (mjData.ncon).
    public let contactCount: Int

    /// A state from its parts; no count is checked (the reader that produced them owns the sizes).
    public init(time: Double, qpos: [Double], qvel: [Double], control: [Double], sensordata: [Double],
                xpos: [Double], xquat: [Double], actuatorForce: [Double], contactCount: Int) {
        self.time = time
        self.qpos = qpos
        self.qvel = qvel
        self.control = control
        self.sensordata = sensordata
        self.xpos = xpos
        self.xquat = xquat
        self.actuatorForce = actuatorForce
        self.contactCount = contactCount
    }

    /// A copy of the current state of `physics`.
    public init(reading physics: any PhysicsReading) {
        self.init(time: physics.time,
                  qpos: physics.qpos,
                  qvel: physics.qvel,
                  control: physics.control,
                  sensordata: physics.sensordata,
                  xpos: physics.xpos,
                  xquat: physics.xquat,
                  actuatorForce: physics.actuatorForce,
                  contactCount: physics.contacts.count)
    }

    /// The names of the fields holding a NaN or an infinity, in declaration sequence; empty when
    /// every number is finite.
    public var nonFiniteFields: [String] {
        var names: [String] = []
        if !time.isFinite { names.append("time") }
        let arrays: [(String, [Double])] = [
            ("qpos", qpos), ("qvel", qvel), ("control", control), ("sensordata", sensordata),
            ("xpos", xpos), ("xquat", xquat), ("actuatorForce", actuatorForce),
        ]
        for (name, values) in arrays where !values.allSatisfy({ $0.isFinite }) {
            names.append(name)
        }
        return names
    }

    /// True when every number of the state is finite.
    public var isFinite: Bool { nonFiniteFields.isEmpty }
}

// MARK: - The model constants

/// The mjModel constants the sense readers and the command decoder read, copied once from the
/// loaded model (field comments of include/mujoco/mjmodel.h at MuJoCo 3.3.0 in brackets). Per-joint
/// arrays have njnt entries, per-actuator arrays nu, per-body arrays nbody, per-sensor arrays
/// nsensor, `cameraNames` ncam, `geomBodyID` ngeom. A name the model does not give is the empty
/// string.
/// `actuatorForceRange*` and `actuatorGainParameter0` are the values at the time of the copy:
/// the body-function modulation rewrites both (env.py:2843-2845).
public struct PhysicsModelTables: Equatable, Codable, Sendable {
    /// The physics timestep in seconds [opt.timestep], 0.005 on the released scene.
    public let timestep: Double
    /// Body names by body id, body 0 the world [mj_id2name, mjOBJ_BODY].
    public let bodyNames: [String]
    /// [body_parentid] "id of body's parent".
    public let bodyParentID: [Int]
    /// [body_mass] kilograms.
    public let bodyMass: [Double]
    /// Joint names by joint id [mj_id2name, mjOBJ_JOINT].
    public let jointNames: [String]
    /// [jnt_type] mjtJoint: 0 free, 1 ball, 2 slide, 3 hinge (mjmodel.h:87-90).
    public let jointType: [Int]
    /// [jnt_bodyid] "id of joint's body".
    public let jointBodyID: [Int]
    /// [jnt_qposadr] "start addr in 'qpos' for joint's data".
    public let jointQposAddress: [Int]
    /// [jnt_dofadr] "start addr in 'qvel' for joint's data".
    public let jointDofAddress: [Int]
    /// [jnt_limited] "does joint have limits".
    public let jointLimited: [Bool]
    /// [jnt_range] column 0, radians for a hinge.
    public let jointRangeLow: [Double]
    /// [jnt_range] column 1.
    public let jointRangeHigh: [Double]
    /// [qpos0] "qpos values at default pose", nq.
    public let qpos0: [Double]
    /// Actuator names by actuator id [mj_id2name, mjOBJ_ACTUATOR], "act:hip_bend" first.
    public let actuatorNames: [String]
    /// [actuator_trntype] mjtTrn, 0 = joint (mjmodel.h:203).
    public let actuatorTransmissionType: [Int]
    /// [actuator_trnid] column 0: the joint id under a joint transmission.
    public let actuatorTransmissionID: [Int]
    /// [actuator_ctrllimited] "is control limited".
    public let actuatorControlLimited: [Bool]
    /// [actuator_ctrlrange] column 0; -1 for every motor of the released scene.
    public let actuatorControlRangeLow: [Double]
    /// [actuator_ctrlrange] column 1; 1 for every motor of the released scene.
    public let actuatorControlRangeHigh: [Double]
    /// [actuator_forcelimited] "is force limited".
    public let actuatorForceLimited: [Bool]
    /// [actuator_forcerange] column 0 (MIMo's action space low, actuation.py:136-138).
    public let actuatorForceRangeLow: [Double]
    /// [actuator_forcerange] column 1 (MIMo's action space high).
    public let actuatorForceRangeHigh: [Double]
    /// [actuator_gaintype] mjtGain, 0 = fixed (mjmodel.h:225).
    public let actuatorGainType: [Int]
    /// [actuator_biastype] mjtBias, 0 = none (mjmodel.h:233).
    public let actuatorBiasType: [Int]
    /// [actuator_gainprm] column 0 of mjNGAIN = 10 (mjmodel.h:39): the fixed force gain.
    public let actuatorGainParameter0: [Double]
    /// [actuator_gear] column 0 of 6: MIMo's maximum motor torque (actuation.py:125).
    public let actuatorGear0: [Double]
    /// Sensor names by sensor id [mj_id2name, mjOBJ_SENSOR], "vestibular_acc" first.
    public let sensorNames: [String]
    /// [sensor_adr] "address in sensor array".
    public let sensorAddress: [Int]
    /// [sensor_dim] "number of scalar outputs".
    public let sensorDimension: [Int]
    /// Camera names by camera id [mj_id2name, mjOBJ_CAMERA]: watch_cam, eye_left, fovea_left,
    /// ring_left, eye_right, fovea_right on the released scene (Python mujoco 3.3.0, November 4, 2026).
    public let cameraNames: [String]
    /// [geom_bodyid] "id of geom's body", ngeom entries (188 on the released scene); empty when
    /// the tables were built by hand without it.
    public let geomBodyID: [Int]
    /// [body_rootid] "id of root above body", nbody entries; empty when the tables were built by
    /// hand without it.
    public let bodyRootID: [Int]

    /// The tables from their parts; every per-joint, per-actuator, per-body and per-sensor array
    /// must have the count of its names array (`bodyRootID` may also be empty), or
    /// `PhysicsError.wrongCount` names the first that does not. `geomBodyID` has no names array
    /// to check against.
    public init(timestep: Double,
                bodyNames: [String], bodyParentID: [Int], bodyMass: [Double],
                jointNames: [String], jointType: [Int], jointBodyID: [Int],
                jointQposAddress: [Int], jointDofAddress: [Int], jointLimited: [Bool],
                jointRangeLow: [Double], jointRangeHigh: [Double],
                qpos0: [Double],
                actuatorNames: [String], actuatorTransmissionType: [Int], actuatorTransmissionID: [Int],
                actuatorControlLimited: [Bool], actuatorControlRangeLow: [Double],
                actuatorControlRangeHigh: [Double], actuatorForceLimited: [Bool],
                actuatorForceRangeLow: [Double], actuatorForceRangeHigh: [Double],
                actuatorGainType: [Int], actuatorBiasType: [Int], actuatorGainParameter0: [Double],
                actuatorGear0: [Double],
                sensorNames: [String], sensorAddress: [Int], sensorDimension: [Int],
                cameraNames: [String],
                geomBodyID: [Int] = [], bodyRootID: [Int] = []) throws {
        let nbody = bodyNames.count
        let njnt = jointNames.count
        let nu = actuatorNames.count
        let nsensor = sensorNames.count
        var checks: [(String, Int, Int)] = [
            ("bodyParentID", nbody, bodyParentID.count), ("bodyMass", nbody, bodyMass.count),
            ("jointType", njnt, jointType.count), ("jointBodyID", njnt, jointBodyID.count),
            ("jointQposAddress", njnt, jointQposAddress.count),
            ("jointDofAddress", njnt, jointDofAddress.count), ("jointLimited", njnt, jointLimited.count),
            ("jointRangeLow", njnt, jointRangeLow.count), ("jointRangeHigh", njnt, jointRangeHigh.count),
            ("actuatorTransmissionType", nu, actuatorTransmissionType.count),
            ("actuatorTransmissionID", nu, actuatorTransmissionID.count),
            ("actuatorControlLimited", nu, actuatorControlLimited.count),
            ("actuatorControlRangeLow", nu, actuatorControlRangeLow.count),
            ("actuatorControlRangeHigh", nu, actuatorControlRangeHigh.count),
            ("actuatorForceLimited", nu, actuatorForceLimited.count),
            ("actuatorForceRangeLow", nu, actuatorForceRangeLow.count),
            ("actuatorForceRangeHigh", nu, actuatorForceRangeHigh.count),
            ("actuatorGainType", nu, actuatorGainType.count), ("actuatorBiasType", nu, actuatorBiasType.count),
            ("actuatorGainParameter0", nu, actuatorGainParameter0.count),
            ("actuatorGear0", nu, actuatorGear0.count),
            ("sensorAddress", nsensor, sensorAddress.count), ("sensorDimension", nsensor, sensorDimension.count),
        ]
        if !bodyRootID.isEmpty {
            checks.append(("bodyRootID", nbody, bodyRootID.count))
        }
        for (name, expected, found) in checks where expected != found {
            throw PhysicsError.wrongCount(name: name, expected: expected, found: found)
        }
        self.timestep = timestep
        self.bodyNames = bodyNames
        self.bodyParentID = bodyParentID
        self.bodyMass = bodyMass
        self.jointNames = jointNames
        self.jointType = jointType
        self.jointBodyID = jointBodyID
        self.jointQposAddress = jointQposAddress
        self.jointDofAddress = jointDofAddress
        self.jointLimited = jointLimited
        self.jointRangeLow = jointRangeLow
        self.jointRangeHigh = jointRangeHigh
        self.qpos0 = qpos0
        self.actuatorNames = actuatorNames
        self.actuatorTransmissionType = actuatorTransmissionType
        self.actuatorTransmissionID = actuatorTransmissionID
        self.actuatorControlLimited = actuatorControlLimited
        self.actuatorControlRangeLow = actuatorControlRangeLow
        self.actuatorControlRangeHigh = actuatorControlRangeHigh
        self.actuatorForceLimited = actuatorForceLimited
        self.actuatorForceRangeLow = actuatorForceRangeLow
        self.actuatorForceRangeHigh = actuatorForceRangeHigh
        self.actuatorGainType = actuatorGainType
        self.actuatorBiasType = actuatorBiasType
        self.actuatorGainParameter0 = actuatorGainParameter0
        self.actuatorGear0 = actuatorGear0
        self.sensorNames = sensorNames
        self.sensorAddress = sensorAddress
        self.sensorDimension = sensorDimension
        self.cameraNames = cameraNames
        self.geomBodyID = geomBodyID
        self.bodyRootID = bodyRootID
    }

    /// nbody.
    public var bodyCount: Int { bodyNames.count }
    /// njnt.
    public var jointCount: Int { jointNames.count }
    /// nu.
    public var actuatorCount: Int { actuatorNames.count }
    /// nsensor.
    public var sensorCount: Int { sensorNames.count }
    /// ngeom as `geomBodyID` gives it; 0 for tables built without it.
    public var geomCount: Int { geomBodyID.count }
}

// MARK: - The protocols beside Physics

/// Gives the loaded model's constants (the model-description protocol that UEMACore's
/// PhysicsReading names, Protocols.swift:380-383). Class-bound like PhysicsReading.
public protocol PhysicsModelDescribing: AnyObject {
    /// The constants of the loaded model, copied now; throws `PhysicsError.notLoaded` before a load.
    func modelTables() throws -> PhysicsModelTables
}

/// The two model writes of the body-function modulation (env.py:2832-2845, the released run's
/// `body_function modulate`): `actuator_gainprm[:, 0] = native_gain * factor` and
/// `actuator_forcerange = native_range * factor`. The caller (WI-37's MotorFunction) keeps the
/// native values and the factors; this protocol only writes the numbers it is given.
public protocol ActuatorParameterWriting: AnyObject {
    /// Writes `values` (nu numbers) into actuator_gainprm column 0.
    func setActuatorGainParameter0(_ values: [Double]) throws

    /// Writes `low` and `high` (nu numbers each) into actuator_forcerange columns 0 and 1.
    func setActuatorForceRange(low: [Double], high: [Double]) throws
}

/// A body's world position under chosen joint angles, computed on a scratch copy so that the
/// state the caller holds is not changed (action_decoder.py:224-238: `scratch = mujoco.MjData(model)`,
/// qpos entries set, `mujoco.mj_forward(model, scratch)`, `scratch.xpos[body_id]` read).
public protocol ScratchKinematicsQuerying: AnyObject {
    /// The world position (3 numbers, metres) of `body` when each joint id in `jointAngles` holds
    /// the given angle (radians, written at the joint's jnt_qposadr) and every other qpos entry holds
    /// its qpos0 value.
    func scratchBodyPosition(ofBody body: Int, withJointAngles jointAngles: [Int: Double]) throws -> [Double]
}
