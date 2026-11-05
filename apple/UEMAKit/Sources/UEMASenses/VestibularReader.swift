// VestibularReader.swift: the vestibular sense of the UEMA Apple port, its 29 raw fields
// (work item WI-27, target B of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html;
// specification REQ-B-4; contract apple/UEMAKit/Specs/vestibular_fields.md sections 1 and 6).
//
// WHAT THIS FILE IS. The port of `read_vestibular_raw` and `vestibular_from_raw`
// (Source/Python/mimo_bridge/vestibular_adapter.py:227-292), of the model indices `_Layout`
// looks up once per world (:146-176), of the support groups (:127-143 over
// touch_adapter.py:41-65) and of the support sum `_support` (:194-209).
//
// INPUT, read once per model step from the state after its last env step (the model reads
// "ONE vector per model step, the last env step's", vestibular_fields.md section 1):
// `PhysicsReading.sensordata` (the head accelerometer `vestibular_acc`, m/s^2, and gyro
// `vestibular_gyro`, rad/s, both head frame), `qpos`, `qvel`, `actuatorForce`, `xmat` (the
// chest's rotation); the contact records of that state (`SupportContactProviding`); the
// fatigue capacities of the shared `ProprioceptionSampling`, advanced for that env step
// before this read (env.py:3133-3139; vestibular_fields.md section 6 point A).
//
// OPERATION (vestibular_adapter.py:268-292; every scale with its line):
//   0-2   head_acc        acc / 9.81                         GRAVITY_MS2 :64
//   3-5   head_gyro       gyro / 2.0                         GYRO_SCALE_RADS :86
//   6-8   neck angles     clip(2 (q - lo) / max(hi - lo, 1e-9) - 1, -1, 1) for
//                         robot:head_swivel, robot:head_tilt, robot:head_tilt_side  :58, :277
//   9-11  neck speeds     qvel[dof] / 1.5                    NECK_VEL_SCALE_RADS :87
//   12-14 neck efforts    f / hi when f >= 0 (hi > 0, else 1); f / (-lo) when f < 0 (lo < 0,
//                         else 1); f = actuator_force of act:head_swivel, act:head_tilt,
//                         act:head_tilt_side; lo, hi the NATIVE force range saved before
//                         fatigue scales it (env.py:2839)                         :59, :236-243
//   15-17 trunk_up        row 2 of the chest's rotation matrix                    :244, :280
//   18-21 support shares  support_n / total, 0 when total = 0; support_n[g] sums max(0,
//                         normal force) of every contact record whose other body is not the
//                         infant's own, on a body of group g (head 0, trunk 1, arms and hands
//                         2, legs and feet 3)                                     :194-209, :281-283
//   22    support_total   total / max(weight_n, 1e-9); weight_n = sum(body_mass[own]) x 9.81
//                         N, 1.0 with no own body                                 :175, :284
//   23    effort_power    sum |actuator_force x actuator_velocity| / 66.0         :88, :260
//   24    effort_holding  sum |actuator_force x gear[0]| / 63.0                   :89, :262
//   25    limit_load      share of motor-driven, limited, non-eye hinge joints with
//                         min(q - lo, hi - q) < 0.035 rad; 0 with no such joint  :69, :162-171, :246-247
//   26    capacity_min    min(cap) of the 10 body-set capacities              :289
//   27    capacity_mean   mean(cap)                                           :290
//   28    valid           1.0 on a live read                                  :291
// actuator_velocity is formed here as gear[0] x qvel[dof of the actuator's joint]: for an
// actuator of a joint transmission on a hinge or slide joint MuJoCo's actuator_moment is
// gear[0] and actuator_velocity = moment x qvel; the layout builder refuses any other
// actuator (`BodySenseError.unsupportedActuator`). Arithmetic is in Double as numpy's float64
// and each field is cast to Float as the float32 vector stores it (:270).
//
// OUTPUT. `SenseReading(sense: .vestibular, primary: [29])`. The tokenizer (work item WI-30,
// `VestibularNormaliser`) drops `valid`, clips the 28 others to -4 ... 4, centres and scales
// them by the checkpoint's frozen mean and variance (vestibular_fields.md sections 2-3).
//
// SUPPORT CONTACTS. `ContactListSupport` builds the records of `BridgeEnv._contacts`
// (env.py:2305-2356) that `_support` reads, from `PhysicsReading.contacts`: a contact counts
// only when one of its two bodies carries skin; it is skipped when |mj_contactForce[0]| <
// 1e-9 N; it gives one record per skin body with normal force |fn| and `otherIsSelf` true when
// the other body is one of the infant's own bodies (env.py:2349; own bodies are those sharing
// the root body of `hip`, env.py:1200-1203).
//
// SOLID. Single responsibility: layout lookup, support records and the field arithmetic are
// separate types or functions. Dependency inversion: the reader takes the layout value, a
// `ProprioceptionSampling` and a `SupportContactProviding` at init.
//
// SOURCES READ BEFORE WRITING (November 4, 2026): Source/Python/mimo_bridge/
// vestibular_adapter.py lines 1-321 (read entire); touch_adapter.py lines 41-65
// (SKIN_REGIONS, region_of); env.py lines 1196-1205 (own bodies), 2305-2356 (`_contacts`),
// 2839 (native force range), 3113-3139 (`_collect` sequence); apple/UEMAKit/Specs/
// vestibular_fields.md (all sections).
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain). The "First build on a Mac" checklist of this work item is in
// ProprioceptionReader.swift's header.

import Foundation
import UEMACore

// MARK: - Support contacts

/// One contact record as `_support` reads it (vestibular_adapter.py:203-208): the skin body
/// it acts on, its normal force in newtons (|fn|) and whether the other body is the infant's.
public struct SupportContact: Equatable, Hashable, Sendable {
    /// The MuJoCo id of the body the record acts on.
    public let bodyId: Int
    /// The normal force, newtons.
    public let normalForce: Double
    /// True when the other body of the contact is one of the infant's own bodies.
    public let otherIsSelf: Bool

    /// A record.
    public init(bodyId: Int, normalForce: Double, otherIsSelf: Bool) {
        self.bodyId = bodyId
        self.normalForce = normalForce
        self.otherIsSelf = otherIsSelf
    }
}

/// The contact records of the current state for the vestibular support fields.
public protocol SupportContactProviding: AnyObject {
    /// The records of the contacts `physics` holds now.
    func supportContacts(_ physics: any PhysicsReading) throws -> [SupportContact]
}

/// The records of `BridgeEnv._contacts` (env.py:2305-2356) built from
/// `PhysicsReading.contacts`, keeping only what `_support` reads.
public final class ContactListSupport: SupportContactProviding {
    /// Contacts whose |normal force| is below this are skipped: 1e-9 N (env.py:2318).
    public static let minimumNormalForce = 1e-9

    private let geomBodyIds: [Int]
    private let skinBodyIds: Set<Int>
    private let ownBodyIds: Set<Int>

    /// A builder over `geomBodyIds` (`geom_bodyid`, one per geom), the bodies that carry
    /// skin (`touch.has_sensors`, the skin layout's bodies of WI-13) and the infant's own
    /// bodies.
    public init(geomBodyIds: [Int], skinBodyIds: Set<Int>, ownBodyIds: Set<Int>) {
        self.geomBodyIds = geomBodyIds
        self.skinBodyIds = skinBodyIds
        self.ownBodyIds = ownBodyIds
    }

    public func supportContacts(_ physics: any PhysicsReading) throws -> [SupportContact] {
        var out: [SupportContact] = []
        for contact in physics.contacts {
            let geom1 = contact.geomIds[0]
            let geom2 = contact.geomIds[1]
            guard geom1 >= 0, geom1 < geomBodyIds.count, geom2 >= 0, geom2 < geomBodyIds.count else {
                continue
            }
            let body1 = geomBodyIds[geom1]
            let body2 = geomBodyIds[geom2]
            let sensing = [body1, body2].filter { skinBodyIds.contains($0) }
            if sensing.isEmpty {
                continue
            }
            let normal = contact.force[0]
            if abs(normal) < ContactListSupport.minimumNormalForce {
                continue
            }
            for body in sensing {
                let other = body == body2 ? body1 : body2
                out.append(SupportContact(bodyId: body, normalForce: abs(normal), otherIsSelf: ownBodyIds.contains(other)))
            }
        }
        return out
    }
}

// MARK: - Layout

/// The model indices and constants the 29 fields need, looked up once per loaded model
/// (`_Layout`, vestibular_adapter.py:146-176).
public struct VestibularLayout: Equatable, Sendable {
    /// The neck joints (`NECK_JOINTS`, vestibular_adapter.py:58).
    public static let neckJointNames = ["robot:head_swivel", "robot:head_tilt", "robot:head_tilt_side"]

    /// The neck actuators (`NECK_ACTUATORS`, vestibular_adapter.py:59).
    public static let neckActuatorNames = ["act:head_swivel", "act:head_tilt", "act:head_tilt_side"]

    /// The accelerometer sensor's name (vestibular_adapter.py:152).
    public static let accelerometerName = "vestibular_acc"

    /// The gyro sensor's name (vestibular_adapter.py:153).
    public static let gyroName = "vestibular_gyro"

    /// The region prefixes of the skin (`SKIN_REGIONS`, touch_adapter.py:41-56), in their
    /// matching sequence.
    public static let skinRegions: [(region: String, prefixes: [String])] = [
        ("head", ["head"]),
        ("trunk", ["chest", "upper_body", "lower_body", "hip"]),
        ("left hand", ["left_hand", "left_ff", "left_mf", "left_rf", "left_lf", "left_th"]),
        ("right hand", ["right_hand", "right_ff", "right_mf", "right_rf", "right_lf", "right_th"]),
        ("left lower arm", ["left_lower_arm"]),
        ("right lower arm", ["right_lower_arm"]),
        ("left upper arm", ["left_upper_arm"]),
        ("right upper arm", ["right_upper_arm"]),
        ("left upper leg", ["left_upper_leg"]),
        ("right upper leg", ["right_upper_leg"]),
        ("left lower leg", ["left_lower_leg"]),
        ("right lower leg", ["right_lower_leg"]),
        ("left foot", ["left_foot", "left_toes", "left_big_toe"]),
        ("right foot", ["right_foot", "right_toes", "right_big_toe"]),
    ]

    /// sensordata indices of the accelerometer's 3 numbers.
    public let accelerometerAddresses: [Int]
    /// sensordata indices of the gyro's 3 numbers.
    public let gyroAddresses: [Int]
    /// qpos indices of the 3 neck joints.
    public let neckQposAddresses: [Int]
    /// qvel indices of the 3 neck joints.
    public let neckDofAddresses: [Int]
    /// `jnt_range` of the 3 neck joints, radians.
    public let neckRanges: [LimitPair]
    /// Actuator ids of the 3 neck actuators.
    public let neckActuators: [Int]
    /// The NATIVE `actuator_forcerange` of the 3 neck actuators, before fatigue scaling.
    public let neckForceRanges: [LimitPair]
    /// The chest body's id.
    public let chestBodyId: Int
    /// gear[0] of every actuator (nu).
    public let actuatorGears: [Double]
    /// qvel index of every actuator's joint (nu).
    public let actuatorDofs: [Int]
    /// qpos indices of the motor-driven, limited, non-eye hinge joints.
    public let limitQposAddresses: [Int]
    /// `jnt_range` of those joints.
    public let limitRanges: [LimitPair]
    /// The infant's weight in newtons: sum(body_mass[own]) x 9.81, 1.0 with no own body.
    public let weightNewtons: Double
    /// The support group of every body (nbody): 0 head, 1 trunk, 2 arms and hands, 3 legs
    /// and feet, -1 none.
    public let supportGroups: [Int]

    /// A layout from its tables; wrong counts throw `BodySenseError.wrongCount`.
    public init(accelerometerAddresses: [Int], gyroAddresses: [Int],
                neckQposAddresses: [Int], neckDofAddresses: [Int], neckRanges: [LimitPair],
                neckActuators: [Int], neckForceRanges: [LimitPair], chestBodyId: Int,
                actuatorGears: [Double], actuatorDofs: [Int],
                limitQposAddresses: [Int], limitRanges: [LimitPair],
                weightNewtons: Double, supportGroups: [Int]) throws {
        let threes: [(String, Int)] = [
            ("accelerometerAddresses", accelerometerAddresses.count), ("gyroAddresses", gyroAddresses.count),
            ("neckQposAddresses", neckQposAddresses.count), ("neckDofAddresses", neckDofAddresses.count),
            ("neckRanges", neckRanges.count), ("neckActuators", neckActuators.count),
            ("neckForceRanges", neckForceRanges.count),
        ]
        for (name, count) in threes where count != 3 {
            throw BodySenseError.wrongCount(type: "VestibularLayout.\(name)", expected: 3, found: count)
        }
        guard actuatorDofs.count == actuatorGears.count else {
            throw BodySenseError.wrongCount(
                type: "VestibularLayout.actuatorDofs", expected: actuatorGears.count, found: actuatorDofs.count)
        }
        guard limitRanges.count == limitQposAddresses.count else {
            throw BodySenseError.wrongCount(
                type: "VestibularLayout.limitRanges", expected: limitQposAddresses.count, found: limitRanges.count)
        }
        self.accelerometerAddresses = accelerometerAddresses
        self.gyroAddresses = gyroAddresses
        self.neckQposAddresses = neckQposAddresses
        self.neckDofAddresses = neckDofAddresses
        self.neckRanges = neckRanges
        self.neckActuators = neckActuators
        self.neckForceRanges = neckForceRanges
        self.chestBodyId = chestBodyId
        self.actuatorGears = actuatorGears
        self.actuatorDofs = actuatorDofs
        self.limitQposAddresses = limitQposAddresses
        self.limitRanges = limitRanges
        self.weightNewtons = weightNewtons
        self.supportGroups = supportGroups
    }

    /// The skin region of a body name (`region_of`, touch_adapter.py:59-65): the first
    /// region with a prefix the name equals or starts with; "" when none.
    public static func region(ofBodyNamed name: String) -> String {
        for entry in skinRegions {
            for prefix in entry.prefixes where name == prefix || name.hasPrefix(prefix) {
                return entry.region
            }
        }
        return ""
    }

    /// The support group of a body name (`_group_of_body`, vestibular_adapter.py:127-143):
    /// head region or a name ending in "_eye" 0; trunk 1; a region naming a hand or an arm 2;
    /// a region naming a leg or a foot 3; else -1.
    public static func supportGroup(ofBodyNamed name: String) -> Int {
        let region = region(ofBodyNamed: name)
        if region == "head" || name.hasSuffix("_eye") {
            return 0
        }
        if region == "trunk" {
            return 1
        }
        if region.contains("hand") || region.contains("arm") {
            return 2
        }
        if region.contains("leg") || region.contains("foot") {
            return 3
        }
        return -1
    }

    /// The infant's own bodies: every body whose `body_rootid` equals that of `hip`
    /// (env.py:1200-1203); empty when the model has no body named hip (env.py:1204-1205).
    public static func ownBodyIds(of model: any MuJoCoModelDescribing) -> Set<Int> {
        guard let hip = model.id(of: .body, named: "hip") else {
            return []
        }
        let root = model.bodyRootId(hip)
        return Set((0 ..< model.sizes.nbody).filter { model.bodyRootId($0) == root })
    }

    /// The layout of a loaded model (vestibular_adapter.py:149-176). `ownBodyIds` are the
    /// infant's bodies (env.py:1200-1203); `nativeForceRanges` the `actuator_forcerange`
    /// rows saved before any fatigue scaling (env.py:2839), one per actuator, nil to read the
    /// model's current rows. An own body id outside 0 ..< nbody throws
    /// `BodySenseError.arrayTooShort` (numpy's `body_mass[own]` raises IndexError there).
    public static func make(from model: any MuJoCoModelDescribing, ownBodyIds: Set<Int>,
                            nativeForceRanges: [LimitPair]? = nil) throws -> VestibularLayout {
        func sensorAddresses(_ name: String) throws -> [Int] {
            let sensor = try model.requireId(of: .sensor, named: name)
            let start = model.sensorAddress(sensor)
            return Array(start ..< start + model.sensorDimension(sensor))
        }
        let accelerometer = try sensorAddresses(accelerometerName)
        let gyro = try sensorAddresses(gyroName)
        let neckJoints = try neckJointNames.map { try model.requireId(of: .joint, named: $0) }
        let neckActuators = try neckActuatorNames.map { try model.requireId(of: .actuator, named: $0) }
        if let native = nativeForceRanges, native.count != model.sizes.nu {
            throw BodySenseError.wrongCount(
                type: "VestibularLayout nativeForceRanges", expected: model.sizes.nu, found: native.count)
        }
        let neckForceRanges = neckActuators.map { actuator in
            nativeForceRanges?[actuator] ?? model.actuatorForceRange(actuator)
        }
        var gears: [Double] = []
        var dofs: [Int] = []
        var limitJoints: [Int] = []
        for actuator in 0 ..< model.sizes.nu {
            let (joint, dof) = try model.scalarJoint(ofActuator: actuator)
            gears.append(model.actuatorGear(actuator))
            dofs.append(dof)
            let jointName = model.name(of: .joint, id: joint) ?? ""
            if model.actuatorTransmissionType(actuator) == .joint,
               model.jointType(joint) == .hinge,
               model.jointIsLimited(joint),
               !jointName.contains("_eye_"),
               !limitJoints.contains(joint) {
                limitJoints.append(joint)
            }
        }
        let own = ownBodyIds.sorted()
        var ownMass = 0.0
        for body in own {
            guard body >= 0, body < model.sizes.nbody else {
                throw BodySenseError.arrayTooShort(array: "body_mass", index: body, count: model.sizes.nbody)
            }
            ownMass += model.bodyMass(body)
        }
        let weight = own.isEmpty ? 1.0 : ownMass * VestibularReader.gravity
        let groups = (0 ..< model.sizes.nbody).map { body in
            supportGroup(ofBodyNamed: model.name(of: .body, id: body) ?? "")
        }
        return try VestibularLayout(
            accelerometerAddresses: accelerometer, gyroAddresses: gyro,
            neckQposAddresses: neckJoints.map { model.jointQposAddress($0) },
            neckDofAddresses: neckJoints.map { model.jointDofAddress($0) },
            neckRanges: neckJoints.map { model.jointRange($0) },
            neckActuators: neckActuators, neckForceRanges: neckForceRanges,
            chestBodyId: try model.requireId(of: .body, named: "chest"),
            actuatorGears: gears, actuatorDofs: dofs,
            limitQposAddresses: limitJoints.map { model.jointQposAddress($0) },
            limitRanges: limitJoints.map { model.jointRange($0) },
            weightNewtons: weight, supportGroups: groups)
    }
}

// MARK: - The reader

/// The vestibular `SenseReader` (specification REQ-B-4): float32 [29] per model step from
/// the state after its last env step.
public final class VestibularReader: SenseReader {
    /// The 29 field names (`VESTIBULAR_FIELDS`, vestibular_adapter.py:40-53).
    public static let fieldNames: [String] = [
        "head_acc_x", "head_acc_y", "head_acc_z",
        "head_gyro_x", "head_gyro_y", "head_gyro_z",
        "neck_swivel", "neck_tilt", "neck_tilt_side",
        "neck_swivel_vel", "neck_tilt_vel", "neck_tilt_side_vel",
        "neck_swivel_effort", "neck_tilt_effort", "neck_tilt_side_effort",
        "trunk_up_x", "trunk_up_y", "trunk_up_z",
        "support_head", "support_trunk", "support_arms", "support_legs",
        "support_total",
        "effort_power", "effort_holding",
        "limit_load",
        "capacity_min", "capacity_mean",
        "valid",
    ]

    /// Fields per reading: 29 (`VESTIBULAR_DIM`, vestibular_adapter.py:54).
    public static let fieldCount = 29

    /// One g: 9.81 m/s^2 (`GRAVITY_MS2`, vestibular_adapter.py:64).
    public static let gravity = 9.81

    /// `GYRO_SCALE_RADS` 2.0 rad/s (vestibular_adapter.py:86).
    public static let gyroScale = 2.0

    /// `NECK_VEL_SCALE_RADS` 1.5 rad/s (vestibular_adapter.py:87).
    public static let neckSpeedScale = 1.5

    /// `EFFORT_POWER_SCALE_W` 66.0 (vestibular_adapter.py:88).
    public static let effortPowerScale = 66.0

    /// `EFFORT_HOLDING_SCALE_NM` 63.0 N m (vestibular_adapter.py:89).
    public static let effortHoldingScale = 63.0

    /// `LIMIT_BAND_RAD` 0.035 rad (vestibular_adapter.py:69).
    public static let limitBand = 0.035

    public let sense: SenseName = .vestibular

    /// The model indices.
    public let layout: VestibularLayout

    private let tracker: ProprioceptionSampling
    private let support: SupportContactProviding

    /// A reader over `layout`, the shared tracker (for the capacities) and the contact
    /// records' provider.
    public init(layout: VestibularLayout, tracker: ProprioceptionSampling, support: SupportContactProviding) {
        self.layout = layout
        self.tracker = tracker
        self.support = support
    }

    /// Advances the shared fatigue model once for this env step.
    public func observeEnvStep(_ physics: any PhysicsReading) throws {
        _ = try tracker.sample(physics)
    }

    public func read(_ physics: any PhysicsReading, at step: ModelStep) throws -> SenseReading {
        let sample = try tracker.sample(physics)
        let contacts = try support.supportContacts(physics)
        let values = try VestibularReader.fields(
            layout: layout,
            sensordata: physics.sensordata, qpos: physics.qpos, qvel: physics.qvel,
            actuatorForce: physics.actuatorForce, xmat: physics.xmat,
            supportForces: VestibularReader.supportForces(layout: layout, contacts: contacts),
            capacities: sample.capacities)
        let primary = try ShapedArray(shape: [VestibularReader.fieldCount], values: values)
        return SenseReading(sense: .vestibular, primary: primary)
    }

    public func reset() {
        tracker.reset()
    }

    /// Newtons per support group (`_support`, vestibular_adapter.py:201-209): max(0, normal
    /// force) of each record whose other body is not the infant's, summed into its body's
    /// group; a body out of the table, or of a group outside 0 ... 3 (-1 is "none"), adds
    /// nothing.
    public static func supportForces(layout: VestibularLayout, contacts: [SupportContact]) -> [Double] {
        var out = [0.0, 0.0, 0.0, 0.0]
        for contact in contacts where !contact.otherIsSelf {
            guard contact.bodyId >= 0, contact.bodyId < layout.supportGroups.count else {
                continue
            }
            let group = layout.supportGroups[contact.bodyId]
            if group >= 0, group < out.count {
                out[group] += max(0, contact.normalForce)
            }
        }
        return out
    }

    /// The 29 fields (`vestibular_from_raw` over `read_vestibular_raw`,
    /// vestibular_adapter.py:227-292); see the header's table.
    public static func fields(layout: VestibularLayout, sensordata: [Double], qpos: [Double], qvel: [Double],
                              actuatorForce: [Double], xmat: [Double], supportForces: [Double],
                              capacities: [Double]) throws -> [Float] {
        func value(_ array: [Double], _ index: Int, _ name: String) throws -> Double {
            guard index >= 0, index < array.count else {
                throw BodySenseError.arrayTooShort(array: name, index: index, count: array.count)
            }
            return array[index]
        }
        guard supportForces.count == 4 else {
            throw BodySenseError.wrongCount(type: "supportForces", expected: 4, found: supportForces.count)
        }
        var v = Array(repeating: Float(0), count: fieldCount)
        for axis in 0 ..< 3 {
            let acceleration = try value(sensordata, layout.accelerometerAddresses[axis], "sensordata")
            let turnRate = try value(sensordata, layout.gyroAddresses[axis], "sensordata")
            v[axis] = Float(acceleration / gravity)
            v[3 + axis] = Float(turnRate / gyroScale)
        }
        for joint in 0 ..< 3 {
            let q = try value(qpos, layout.neckQposAddresses[joint], "qpos")
            let lo = layout.neckRanges[joint].lower
            let hi = layout.neckRanges[joint].upper
            let angle = 2 * (q - lo) / max(hi - lo, 1e-9) - 1
            v[6 + joint] = Float(min(max(angle, -1), 1))
            let neckSpeed = try value(qvel, layout.neckDofAddresses[joint], "qvel")
            v[9 + joint] = Float(neckSpeed / neckSpeedScale)
            let f = try value(actuatorForce, layout.neckActuators[joint], "actuator_force")
            let forceLo = layout.neckForceRanges[joint].lower
            let forceHi = layout.neckForceRanges[joint].upper
            let effort = f >= 0 ? f / (forceHi > 0 ? forceHi : 1) : f / (forceLo < 0 ? -forceLo : 1)
            v[12 + joint] = Float(effort)
        }
        for axis in 0 ..< 3 {
            let up = try value(xmat, 9 * layout.chestBodyId + 6 + axis, "xmat")
            v[15 + axis] = Float(up)
        }
        let total = supportForces.reduce(0, +)
        if total > 0 {
            for group in 0 ..< 4 {
                v[18 + group] = Float(supportForces[group] / total)
            }
        }
        v[22] = Float(total / max(layout.weightNewtons, 1e-9))
        var power = 0.0
        var holding = 0.0
        for actuator in 0 ..< layout.actuatorGears.count {
            let f = try value(actuatorForce, actuator, "actuator_force")
            let gear = layout.actuatorGears[actuator]
            let jointSpeed = try value(qvel, layout.actuatorDofs[actuator], "qvel")
            let velocity = gear * jointSpeed
            power += abs(f * velocity)
            holding += abs(f * gear)
        }
        v[23] = Float(power / effortPowerScale)
        v[24] = Float(holding / effortHoldingScale)
        var near = 0
        for (index, address) in layout.limitQposAddresses.enumerated() {
            let q = try value(qpos, address, "qpos")
            let range = layout.limitRanges[index]
            if min(q - range.lower, range.upper - q) < limitBand {
                near += 1
            }
        }
        let count = layout.limitQposAddresses.count
        v[25] = count == 0 ? 0 : Float(Double(near) / Double(count))
        v[26] = capacities.isEmpty ? 1 : Float(capacities.min() ?? 1)
        v[27] = capacities.isEmpty ? 1 : Float(capacities.reduce(0, +) / Double(capacities.count))
        v[28] = 1
        return v
    }
}
