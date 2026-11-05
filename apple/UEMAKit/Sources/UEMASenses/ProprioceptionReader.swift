// ProprioceptionReader.swift: the proprioception sense of the UEMA Apple port and the body
// state it shares with the motor and vestibular readers (work item WI-27, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-4; contract
// apple/UEMAKit/Specs/proprio_rows.md sections 1 to 4).
//
// WHAT THIS FILE IS. The port of `ProprioAdapter.read(dt)` and `FatigueModel`
// (Source/Python/mimo_bridge/proprio_adapter.py:71-120, :149-246) for the released body
// layout `trunk_v1` (26 rows; mimo_bridge/body_layout.py:6-14), split into one type per
// responsibility:
//   `MuJoCoModelDescribing`  the time-invariant MuJoCo model constants the three readers
//                            index with (joint addresses and ranges, actuator transmissions
//                            and gears, body names and masses, sensor addresses).
//   `ModelTablesDescription` that protocol over UEMAPhysics's `PhysicsModelTables`
//                            (PhysicsState.swift:156-302, work item WI-25), the physics'
//                            `PhysicsSizes` and `geom_bodyid`; `body_rootid` computed from
//                            `body_parentid`.
//   `BodySet`               the 10 body sets (lem/model/body_sets.py:31-41) and the
//                            body-name rule `body_set_for_mimo_body` (proprio_adapter.py:48-68).
//   `ProprioceptionLayout`   the 26 row bodies, each row's body set, each actuator's body set,
//                            degree of freedom and gear (proprio_adapter.py:162-180).
//   `FatigueModel`           capacity and accumulated effort per body set
//                            (proprio_adapter.py:105-120).
//   `BodyStateTracker`       advances the fatigue model once per env step and computes the
//                            bones, energy and body-set arrays (proprio_adapter.py:184-243).
//   `ProprioceptionReader`   the `SenseReader` of the proprioception sense: [26, 12].
//
// INPUT, per env step (0.010 s, ModelStep.envStepSeconds): `PhysicsReading.xquat` (nbody x 4,
// w x y z), `xpos` (nbody x 3, metres), `bodyVelocity(bodyId:)` (mj_objectVelocity, flg_local
// 0: rotation 3 then linear 3, world frame, at the body origin; m/s), `actuatorForce` (nu,
// MIMo's motor output before its gear) and `qvel` (nv).
//
// OPERATION (proprio_adapter.py line by line):
//  1. effort per body set (:184-199), effort_power `no_gear` (the released setting,
//     proprio_rows.md section 3): p[a] = |actuator_force[a] x qvel[dof of a's joint]|, summed
//     into the set of the body that carries a's joint; an actuator of no set is dropped.
//     `with_gear` multiplies p[a] by gear[a] (:190-191).
//  2. fatigue step (:112-116): cap += dt (0.05 (1 - cap) - 0.02 effort / 1.0); cap clipped to
//     0 ... 1; accum += effort dt. Starts at cap 1, accum 0 (:106-107).
//  3. bones float32 [26, 11] per row i with body b (:205-212): 0-3 xquat[b] reordered to
//     x y z w; 4-6 xpos[b] x 100 (cm); 7-9 linear velocity x 100 (cm/s); 10 cap[set of row i].
//  4. the gaze row (row 23) quaternion becomes conj(head) x eye in x y z w, computed in
//     float32 from the float32 head row (row 12) and eye row (:223-230).
//  5. energy float32 [3] (:238-241): ph_work = sum(effort) dt; pcc_hold = sum |actuator_force|
//     dt 0.01; total = ph_work + pcc_hold. body_sets float32 [10, 2] = (cap, accum) (:242).
//
// OUTPUT. `ProprioceptionReader.read` returns `SenseReading(sense: .proprioception, primary:
// [26, 12])`: columns 0-10 are the bones of step 3 and 4 in their physical units (cm, cm/s),
// column 11 is the row index 0 ... 25. The tokenizer (work item WI-30) applies the divisors of
// proprio_rows.md section 4: position minus row 0's position over 200 cm
// (`position_range_cm`, tokenizer/config.py:263), velocity over 500 cm/s
// (`velocity_range_cms`, config.py:264), row index over 25 (`proprio_index_range`,
// config.py:271), then flattens to 312. The tracker's `ProprioceptionSample` also feeds the
// motor reader (MotorReader.swift: achieved rows, energy, capability) and the vestibular
// reader (VestibularReader.swift: capacity_min, capacity_mean).
//
// ONE ADVANCE PER ENV STEP, WHOEVER ASKS FIRST. The Python adapter is read once per env step
// in `_collect` (env.py:3133) and the vestibular vector is read after it, so its capacities
// are that env step's (vestibular_fields.md section 6 point A). Here the three readers share
// one `BodyStateTracker` and each calls `sample(_:)` from `observeEnvStep`; the tracker
// advances only when `PhysicsReading.time` differs from the time of its last advance, so the
// fatigue model steps exactly once per env step whichever reader the session calls first.
//
// AN ERROR LEAVES THE FATIGUE STATE UNCHANGED. The tracker steps a copy of the fatigue model,
// builds the rows from the copy and stores the copy only after every array read succeeded, so
// a thrown `BodySenseError` (an array shorter than an index) does not advance the fatigue model
// and a retry within the same env step advances it once.
//
// SOLID. Single responsibility per type (listed above). Liskov: every reader honours
// `SenseReader` (observe, read, reset) and adds no precondition to it. Dependency inversion:
// readers take a `ProprioceptionSampling` and a `PhysicsReading`; layouts are built from a
// `MuJoCoModelDescribing`; nothing here constructs a physics type (`ModelTablesDescription`
// takes the tables a physics already copied).
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave:
// - Source/Python/mimo_bridge/proprio_adapter.py lines 1-246 (read entire): BONE_MAP :27-41,
//   body_set_for_mimo_body :48-68, FatigueModel :105-120, effort_per_set :184-199, read
//   :201-243.
// - Source/Python/lem/model/body_sets.py lines 31-41 (BodySet ordinals), 82-101
//   (HANDLE_BODY_SET), 52-75 (HANDLE_NAMES).
// - Source/Python/mimo_bridge/body_layout.py lines 6-14: TRUNK_NAMES ('lower_trunk',
//   'chest'), TRUNK_BODIES lower_trunk -> 'lower_body', chest -> 'chest'.
// - Source/Python/mimo_bridge/env.py lines 3113-3139 (`_collect`): pooler, contacts, then
//   `pro = self.proprio.read(self.dt)`, then `vest = read_vestibular(self, contacts)`.
// - Source/Python/mimo_bridge/datasource.py lines 119-141: the [26, 12] normalised rows (the
//   tokenizer's step, not this reader's).
// - apple/UEMAKit/Specs/proprio_rows.md sections 1-5; apple/UEMAKit/Sources/UEMACore/
//   Protocols.swift (`PhysicsReading`, `SenseReader`, `SenseReading`: "proprioception
//   [26, 12]").
// - MuJoCo enums (mjtJoint: mjJNT_FREE 0, mjJNT_BALL 1, mjJNT_SLIDE 2, mjJNT_HINGE 3; mjtTrn:
//   mjTRN_JOINT 0, mjTRN_JOINTINPARENT 1, mjTRN_SLIDERCRANK 2, mjTRN_TENDON 3, mjTRN_SITE 4,
//   mjTRN_BODY 5), from MuJoCo's mjmodel.h as known to the writer; not fetched in this item.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1).
//
// FIRST BUILD ON A MAC (this work item's acceptance: "on a fake Physics with known arrays the
// rows equal hand-computed rows; counts 26, 29, 26 x 10"):
//  1. Run the package steps of Package.swift's header first (MuJoCo XCFramework in place,
//     `swift package resolve` exit 0).
//  2. `cd apple/UEMAKit && xcodebuild -scheme UEMAKit build -destination
//     'platform=macOS'` : exit 0. Constructs to check first if this fails: the protocol
//     `MuJoCoModelDescribing` returning `LimitPair` values; `public let sense: SenseName =
//     .proprioception` satisfying the protocol's `{ get }` requirement; `final class` readers
//     implementing the `SenseReader` extension defaults; `Set<Int>` parameters; `import
//     UEMAPhysics` in this file (for `PhysicsModelTables`; a compile error inside UEMAPhysics
//     itself is WI-25's to correct first); `while current != 0, steps <= count` in
//     `ModelTablesDescription.rootIds(parents:)`.
//  3. `xcodebuild -scheme UEMAKit test -destination 'platform=macOS'
//     -only-testing:UEMASensesTests/BodySenseReaderTests` : exit 0 with every test passing
//     (22 tests in BodySenseReaderTests.swift: counts 26 x 12, 26 x 10 with 24 auxiliary,
//     29; the hand-computed rows, fatigue, energy, gaze row, vestibular fields against the
//     Python reference's numbers, support records, layouts, one advance per env step, no
//     fatigue advance on a thrown error, the error cases, the `ModelTablesDescription`
//     adapter). The test file imports UEMAPhysics to build a `PhysicsModelTables`.
//  4. If a fixture of the released checkpoint is wanted beside the hand-computed rows, the
//     WI-14 exporter's state (qpos, qvel, fatigue cap and accum) on the real scene is the
//     next check; it is not part of this item's acceptance.
//  5. Record each step's command, exit code and clock time (AM/PM Pacific) in
//     apple/FIRST_BUILD_RECORD.md (WI-41).

import Foundation
import UEMACore
import UEMAPhysics

// MARK: - Errors

/// The errors of the body sense readers (proprioception, motor, vestibular): a MuJoCo name
/// the model lacks, an array shorter than an index needs, a count that does not match a fixed
/// size, or an actuator whose transmission the readers cannot turn into a joint speed.
public enum BodySenseError: Error, Equatable, Sendable {
    /// The model has no object of `kind` named `name`.
    case missingName(kind: MuJoCoObjectKind, name: String)
    /// `array` has `count` entries; index `index` was needed.
    case arrayTooShort(array: String, index: Int, count: Int)
    /// `type` was given `found` entries where it holds exactly `expected`.
    case wrongCount(type: String, expected: Int, found: Int)
    /// Actuator `actuator` drives something other than one hinge or slide joint.
    case unsupportedActuator(actuator: Int)
    /// Entry `index` of the model table `table` holds `value`, which is not one of the values
    /// that table may hold (an unknown mjtJoint or mjtTrn value, or a body whose parent chain
    /// never reaches the world body).
    case invalidTableValue(table: String, index: Int, value: Int)
}

// MARK: - The model description

/// MuJoCo's joint types (mjtJoint), by their C enum values.
public enum MuJoCoJointType: Int, Equatable, Sendable {
    case free = 0
    case ball = 1
    case slide = 2
    case hinge = 3
}

/// MuJoCo's actuator transmission types (mjtTrn), by their C enum values.
public enum MuJoCoTransmissionType: Int, Equatable, Sendable {
    case joint = 0
    case jointInParent = 1
    case sliderCrank = 2
    case tendon = 3
    case site = 4
    case body = 5
}

/// A lower and an upper bound, as MuJoCo stores `jnt_range` and `actuator_forcerange`
/// (two numbers per row). The upper bound may be below the lower one in a malformed model;
/// nothing here assumes the order.
public struct LimitPair: Equatable, Hashable, Codable, Sendable {
    /// The first number of the row.
    public let lower: Double
    /// The second number of the row.
    public let upper: Double

    /// A pair of bounds.
    public init(lower: Double, upper: Double) {
        self.lower = lower
        self.upper = upper
    }
}

/// The time-invariant constants of the loaded MuJoCo model that the body sense readers
/// index with, each named after the mjModel array it reads. `ModelTablesDescription` (below)
/// adopts it over the `PhysicsModelTables` that UEMAPhysics's `MuJoCoPhysics.modelTables()`
/// copies (work item WI-25); tests adopt it with hand-written tables. Ids are MuJoCo's
/// 0-based ids.
public protocol MuJoCoModelDescribing {
    /// The sizes of the loaded model (nbody, njnt, nu, ngeom and the rest).
    var sizes: PhysicsSizes { get }

    /// The id of the named object of `kind`; nil when the model has no such name
    /// (mj_name2id).
    func id(of kind: MuJoCoObjectKind, named name: String) -> Int?

    /// The name of object `id` of `kind`; nil or "" for an unnamed object (mj_id2name).
    func name(of kind: MuJoCoObjectKind, id: Int) -> String?

    /// `jnt_bodyid[joint]`: the body the joint belongs to.
    func jointBodyId(_ joint: Int) -> Int

    /// `jnt_type[joint]`.
    func jointType(_ joint: Int) -> MuJoCoJointType

    /// `jnt_limited[joint]` as a boolean.
    func jointIsLimited(_ joint: Int) -> Bool

    /// `jnt_range[joint]`, radians for a hinge.
    func jointRange(_ joint: Int) -> LimitPair

    /// `jnt_qposadr[joint]`: the joint's first index into qpos.
    func jointQposAddress(_ joint: Int) -> Int

    /// `jnt_dofadr[joint]`: the joint's first index into qvel.
    func jointDofAddress(_ joint: Int) -> Int

    /// `actuator_trntype[actuator]`.
    func actuatorTransmissionType(_ actuator: Int) -> MuJoCoTransmissionType

    /// `actuator_trnid[actuator, 0]`: the joint (for a joint transmission) the actuator drives.
    func actuatorTransmissionTarget(_ actuator: Int) -> Int

    /// `actuator_gear[actuator, 0]`: joint torque per unit of actuator force.
    func actuatorGear(_ actuator: Int) -> Double

    /// `actuator_forcerange[actuator]` as the model holds it now.
    func actuatorForceRange(_ actuator: Int) -> LimitPair

    /// `body_mass[body]`, kilograms.
    func bodyMass(_ body: Int) -> Double

    /// `body_rootid[body]`: the ancestor that is a direct child of the world.
    func bodyRootId(_ body: Int) -> Int

    /// `geom_bodyid[geom]`.
    func geomBodyId(_ geom: Int) -> Int

    /// `sensor_adr[sensor]`: the sensor's first index into sensordata.
    func sensorAddress(_ sensor: Int) -> Int

    /// `sensor_dim[sensor]`.
    func sensorDimension(_ sensor: Int) -> Int
}

extension MuJoCoModelDescribing {
    /// The id of the named object of `kind`; a missing name throws
    /// `BodySenseError.missingName`.
    public func requireId(of kind: MuJoCoObjectKind, named name: String) throws -> Int {
        guard let id = id(of: kind, named: name) else {
            throw BodySenseError.missingName(kind: kind, name: name)
        }
        return id
    }

    /// The joint an actuator drives and that joint's first qvel index, for an actuator of a
    /// joint transmission (mjTRN_JOINT or mjTRN_JOINTINPARENT) on a hinge or slide joint;
    /// any other actuator throws `BodySenseError.unsupportedActuator`. For such an actuator
    /// MuJoCo's actuator_moment is gear[0] and actuator_velocity = gear[0] x qvel[dof].
    public func scalarJoint(ofActuator actuator: Int) throws -> (joint: Int, dof: Int) {
        let type = actuatorTransmissionType(actuator)
        guard type == .joint || type == .jointInParent else {
            throw BodySenseError.unsupportedActuator(actuator: actuator)
        }
        let joint = actuatorTransmissionTarget(actuator)
        guard joint >= 0, joint < sizes.njnt else {
            throw BodySenseError.unsupportedActuator(actuator: actuator)
        }
        let kind = self.jointType(joint)
        guard kind == .hinge || kind == .slide else {
            throw BodySenseError.unsupportedActuator(actuator: actuator)
        }
        return (joint, jointDofAddress(joint))
    }
}

// MARK: - The model description over UEMAPhysics's tables

/// `MuJoCoModelDescribing` over the constants UEMAPhysics copies from mjModel
/// (`PhysicsModelTables`, PhysicsState.swift:156-302, work item WI-25), the sizes of the same
/// physics (`PhysicsReading.sizes`) and `geom_bodyid` (ngeom entries, the array the tactile
/// module's `ContactModelTables.geomBodyIds` also holds; empty when no reader here needs it).
/// `body_rootid` is not in the tables, so it is computed from `body_parentid` by its mjmodel.h
/// definition, "ancestor that is direct child of world": body 0 (the world) has root 0, a body
/// whose parent is the world is its own root, any other body takes its parent's root.
public struct ModelTablesDescription: MuJoCoModelDescribing {
    /// The sizes of the physics the tables were copied from.
    public let sizes: PhysicsSizes

    /// The copied mjModel constants.
    public let tables: PhysicsModelTables

    private let geomBodyIds: [Int]
    private let bodyRoots: [Int]
    private let jointTypes: [MuJoCoJointType]
    private let transmissionTypes: [MuJoCoTransmissionType]

    /// A description of `tables`. Throws `BodySenseError.wrongCount` when nbody, njnt or nu of
    /// `sizes` differ from the tables' counts or when a non-empty `geomBodyIds` is not ngeom
    /// long, and `BodySenseError.invalidTableValue` for an unknown jnt_type or actuator_trntype
    /// value or a parent chain that never reaches the world body.
    public init(tables: PhysicsModelTables, sizes: PhysicsSizes, geomBodyIds: [Int] = []) throws {
        let counts: [(String, Int, Int)] = [
            ("PhysicsModelTables.bodyNames", sizes.nbody, tables.bodyCount),
            ("PhysicsModelTables.jointNames", sizes.njnt, tables.jointCount),
            ("PhysicsModelTables.actuatorNames", sizes.nu, tables.actuatorCount),
        ]
        for (name, expected, found) in counts where expected != found {
            throw BodySenseError.wrongCount(type: name, expected: expected, found: found)
        }
        if !geomBodyIds.isEmpty, geomBodyIds.count != sizes.ngeom {
            throw BodySenseError.wrongCount(
                type: "ModelTablesDescription.geomBodyIds", expected: sizes.ngeom, found: geomBodyIds.count)
        }
        var joints: [MuJoCoJointType] = []
        for (joint, raw) in tables.jointType.enumerated() {
            guard let type = MuJoCoJointType(rawValue: raw) else {
                throw BodySenseError.invalidTableValue(table: "jnt_type", index: joint, value: raw)
            }
            joints.append(type)
        }
        var transmissions: [MuJoCoTransmissionType] = []
        for (actuator, raw) in tables.actuatorTransmissionType.enumerated() {
            guard let type = MuJoCoTransmissionType(rawValue: raw) else {
                throw BodySenseError.invalidTableValue(table: "actuator_trntype", index: actuator, value: raw)
            }
            transmissions.append(type)
        }
        self.sizes = sizes
        self.tables = tables
        self.geomBodyIds = geomBodyIds
        self.bodyRoots = try ModelTablesDescription.rootIds(parents: tables.bodyParentID)
        self.jointTypes = joints
        self.transmissionTypes = transmissions
    }

    /// `body_rootid` from `body_parentid`: each body climbs its parents until the next parent
    /// is the world body 0; body 0 has root 0. A parent outside the table or a chain longer
    /// than the body count throws `BodySenseError.invalidTableValue`.
    public static func rootIds(parents: [Int]) throws -> [Int] {
        let count = parents.count
        var roots = Array(repeating: 0, count: count)
        for body in 0 ..< count {
            var current = body
            var steps = 0
            while current != 0, steps <= count {
                let parent = parents[current]
                guard parent >= 0, parent < count else {
                    throw BodySenseError.invalidTableValue(table: "body_parentid", index: current, value: parent)
                }
                if parent == 0 {
                    break
                }
                current = parent
                steps += 1
            }
            if steps > count {
                throw BodySenseError.invalidTableValue(table: "body_parentid", index: body, value: current)
            }
            roots[body] = current
        }
        return roots
    }

    private func names(of kind: MuJoCoObjectKind) -> [String] {
        switch kind {
        case .body: return tables.bodyNames
        case .joint: return tables.jointNames
        case .actuator: return tables.actuatorNames
        case .sensor: return tables.sensorNames
        case .camera: return tables.cameraNames
        case .geom: return []
        }
    }

    public func id(of kind: MuJoCoObjectKind, named name: String) -> Int? {
        guard !name.isEmpty else {
            return nil
        }
        return names(of: kind).firstIndex(of: name)
    }

    public func name(of kind: MuJoCoObjectKind, id: Int) -> String? {
        let all = names(of: kind)
        guard id >= 0, id < all.count else {
            return nil
        }
        return all[id]
    }

    public func jointBodyId(_ joint: Int) -> Int { tables.jointBodyID[joint] }

    public func jointType(_ joint: Int) -> MuJoCoJointType { jointTypes[joint] }

    public func jointIsLimited(_ joint: Int) -> Bool { tables.jointLimited[joint] }

    public func jointRange(_ joint: Int) -> LimitPair {
        LimitPair(lower: tables.jointRangeLow[joint], upper: tables.jointRangeHigh[joint])
    }

    public func jointQposAddress(_ joint: Int) -> Int { tables.jointQposAddress[joint] }

    public func jointDofAddress(_ joint: Int) -> Int { tables.jointDofAddress[joint] }

    public func actuatorTransmissionType(_ actuator: Int) -> MuJoCoTransmissionType { transmissionTypes[actuator] }

    public func actuatorTransmissionTarget(_ actuator: Int) -> Int { tables.actuatorTransmissionID[actuator] }

    public func actuatorGear(_ actuator: Int) -> Double { tables.actuatorGear0[actuator] }

    public func actuatorForceRange(_ actuator: Int) -> LimitPair {
        LimitPair(lower: tables.actuatorForceRangeLow[actuator], upper: tables.actuatorForceRangeHigh[actuator])
    }

    public func bodyMass(_ body: Int) -> Double { tables.bodyMass[body] }

    public func bodyRootId(_ body: Int) -> Int { bodyRoots[body] }

    /// `geom_bodyid[geom]`; -1 for a geom id outside the injected table (or a flex id -1).
    public func geomBodyId(_ geom: Int) -> Int {
        guard geom >= 0, geom < geomBodyIds.count else {
            return -1
        }
        return geomBodyIds[geom]
    }

    public func sensorAddress(_ sensor: Int) -> Int { tables.sensorAddress[sensor] }

    public func sensorDimension(_ sensor: Int) -> Int { tables.sensorDimension[sensor] }
}

// MARK: - Body sets

/// The 10 body sets of the model, by their ordinal (`BodySet`, lem/model/body_sets.py:31-41).
/// The ordinal is the index into the fatigue model's capacity and accumulated-effort arrays
/// and into the motor capability vector (body_sets.py:11-15).
public enum BodySet: Int, CaseIterable, Codable, Hashable, Sendable {
    case armL = 0
    case armR = 1
    case handL = 2
    case handR = 3
    case spine = 4
    case head = 5
    case legL = 6
    case legR = 7
    case footL = 8
    case footR = 9

    /// The number of body sets: 10 (`NUM_BODY_SETS`, body_sets.py:44).
    public static let count = 10

    /// The body set of each of the 26 rows of `trunk_v1`: `HANDLE_BODY_SET`
    /// (body_sets.py:82-101) for the 24 handles, then Spine for lower_trunk and chest
    /// (proprio_adapter.py:167-168).
    public static let ofRows: [BodySet] = [
        .armL, .handL, .handL, .handL, .handL, .handL,
        .armR, .handR, .handR, .handR, .handR, .handR,
        .head,
        .armL, .armL,
        .armR, .armR,
        .legL, .legL,
        .legR, .legR,
        .footL, .footR,
        .head,
        .spine, .spine,
    ]

    /// The body set a MIMo body belongs to, nil for an untracked body: the port of
    /// `body_set_for_mimo_body` (proprio_adapter.py:48-68). A "left_" or "right_" prefix gives
    /// the side and the rest of the name after the first underscore is the base; with a side:
    /// base "hand" or ending in distal, middle, knuckle, metacarpal, thbase or thhub is the
    /// hand; upper_arm, lower_arm the arm; upper_leg, lower_leg the leg; foot, toes, big_toe
    /// the foot; eye the head; anything else nil. Without a side: "head" is the head; hip,
    /// lower_body, upper_body, chest the spine; anything else nil.
    public static func of(mujocoBodyNamed name: String) -> BodySet? {
        let isLeft = name.hasPrefix("left_")
        let isRight = name.hasPrefix("right_")
        if isLeft || isRight {
            let base: String
            if let underscore = name.firstIndex(of: "_") {
                base = String(name[name.index(after: underscore)...])
            } else {
                base = name
            }
            let handSuffixes = ["distal", "middle", "knuckle", "metacarpal", "thbase", "thhub"]
            if base == "hand" || handSuffixes.contains(where: { base.hasSuffix($0) }) {
                return isLeft ? .handL : .handR
            }
            if base == "upper_arm" || base == "lower_arm" {
                return isLeft ? .armL : .armR
            }
            if base == "upper_leg" || base == "lower_leg" {
                return isLeft ? .legL : .legR
            }
            if base == "foot" || base == "toes" || base == "big_toe" {
                return isLeft ? .footL : .footR
            }
            if base == "eye" {
                return .head
            }
            return nil
        }
        if name == "head" {
            return .head
        }
        if ["hip", "lower_body", "upper_body", "chest"].contains(name) {
            return .spine
        }
        return nil
    }
}

// MARK: - Effort formula

/// How the effort that drains the fatigue model is computed (`EFFORT_POWER_MODES`,
/// proprio_adapter.py:131-146). The released run uses `noGear` (proprio_rows.md section 3).
public enum EffortPowerMode: String, Codable, Hashable, Sendable {
    /// |actuator force x joint speed| per actuator (proprio_adapter.py:193).
    case noGear = "no_gear"
    /// |actuator force x gear x joint speed| per actuator, watts (proprio_adapter.py:191).
    case withGear = "with_gear"
}

// MARK: - Layout

/// The model indices the proprioception rows need, looked up once per loaded model
/// (proprio_adapter.py:162-180): the body of each of the 26 rows, and for each of the nu
/// actuators its body set (nil for none), its joint's qvel index and its gear.
public struct ProprioceptionLayout: Equatable, Sendable {
    /// Rows per reading: 26 (`trunk_v1`, body_layout.py:13-14).
    public static let rowCount = BodyHandle.count

    /// The MuJoCo body name of each row: `BONE_MAP` (proprio_adapter.py:27-41; gaze ->
    /// left_eye) then `TRUNK_BODIES` (body_layout.py:7).
    public static let bodyNames: [String] = [
        "left_hand", "left_thdistal", "left_ffdistal", "left_mfdistal", "left_rfdistal", "left_lfdistal",
        "right_hand", "right_thdistal", "right_ffdistal", "right_mfdistal", "right_rfdistal", "right_lfdistal",
        "head",
        "left_upper_arm", "left_lower_arm",
        "right_upper_arm", "right_lower_arm",
        "left_upper_leg", "left_lower_leg",
        "right_upper_leg", "right_lower_leg",
        "left_foot", "right_foot",
        "left_eye",
        "lower_body", "chest",
    ]

    /// The head row, 12, whose quaternion the gaze row is made relative to
    /// (proprio_adapter.py:223).
    public static let headRow = BodyHandle.head.rawValue

    /// The gaze row, 23 (proprio_adapter.py:223).
    public static let gazeRow = BodyHandle.gaze.rawValue

    /// The MuJoCo body id of each of the 26 rows.
    public let bodyIds: [Int]

    /// The body set of each actuator's joint's body; nil drops the actuator from the effort.
    public let actuatorBodySets: [BodySet?]

    /// The qvel index of each actuator's joint (`jnt_dofadr[actuator_trnid[a, 0]]`,
    /// proprio_adapter.py:177).
    public let actuatorDofs: [Int]

    /// Each actuator's gear[0] (proprio_adapter.py:180), used by `EffortPowerMode.withGear`.
    public let actuatorGears: [Double]

    /// A layout from its tables; 26 body ids and equal actuator counts are required, else
    /// `BodySenseError.wrongCount`.
    public init(bodyIds: [Int], actuatorBodySets: [BodySet?], actuatorDofs: [Int], actuatorGears: [Double]) throws {
        guard bodyIds.count == ProprioceptionLayout.rowCount else {
            throw BodySenseError.wrongCount(
                type: "ProprioceptionLayout.bodyIds", expected: ProprioceptionLayout.rowCount, found: bodyIds.count)
        }
        guard actuatorDofs.count == actuatorBodySets.count else {
            throw BodySenseError.wrongCount(
                type: "ProprioceptionLayout.actuatorDofs", expected: actuatorBodySets.count, found: actuatorDofs.count)
        }
        guard actuatorGears.count == actuatorBodySets.count else {
            throw BodySenseError.wrongCount(
                type: "ProprioceptionLayout.actuatorGears", expected: actuatorBodySets.count, found: actuatorGears.count)
        }
        self.bodyIds = bodyIds
        self.actuatorBodySets = actuatorBodySets
        self.actuatorDofs = actuatorDofs
        self.actuatorGears = actuatorGears
    }

    /// The layout of a loaded model (proprio_adapter.py:162-180): row bodies by
    /// `bodyNames`; each actuator's set from its joint's body name through
    /// `BodySet.of(mujocoBodyNamed:)`, its qvel index and its gear.
    public static func make(from model: any MuJoCoModelDescribing) throws -> ProprioceptionLayout {
        let bodyIds = try bodyNames.map { try model.requireId(of: .body, named: $0) }
        var sets: [BodySet?] = []
        var dofs: [Int] = []
        var gears: [Double] = []
        for actuator in 0 ..< model.sizes.nu {
            let (joint, dof) = try model.scalarJoint(ofActuator: actuator)
            let body = model.jointBodyId(joint)
            let bodyName = model.name(of: .body, id: body) ?? ""
            sets.append(BodySet.of(mujocoBodyNamed: bodyName))
            dofs.append(dof)
            gears.append(model.actuatorGear(actuator))
        }
        return try ProprioceptionLayout(bodyIds: bodyIds, actuatorBodySets: sets, actuatorDofs: dofs, actuatorGears: gears)
    }
}

// MARK: - Fatigue

/// Capacity (0 exhausted to 1 fresh) and accumulated effort per body set, driven by
/// mechanical effort: the port of `FatigueModel` (proprio_adapter.py:71-120).
/// d cap / dt = recover (1 - cap) - drain effort / p_ref.
public struct FatigueModel: Equatable, Codable, Sendable {
    /// `recover` 0.05 per second (proprio_adapter.py:105).
    public static let recoverPerSecond = 0.05

    /// `drain` 0.02 per second per unit of effort (proprio_adapter.py:105).
    public static let drainPerSecond = 0.02

    /// `p_ref` 1.0, the effort unit (proprio_adapter.py:105).
    public static let referenceEffort = 1.0

    /// Capacity per body set, 0 ... 1, by `BodySet` ordinal.
    public private(set) var capacities: [Double]

    /// Accumulated effort x seconds per body set, never reset by a step.
    public private(set) var accumulated: [Double]

    /// A fresh model: every capacity 1, every accumulated effort 0 (proprio_adapter.py:106-107).
    public init() {
        capacities = Array(repeating: 1, count: BodySet.count)
        accumulated = Array(repeating: 0, count: BodySet.count)
    }

    /// A model restored from a checkpoint's `cap` and `accum` (proprio_rows.md section 3: "the
    /// fixture starts from the checkpoint's cap and accum"); counts other than 10 throw.
    public init(capacities: [Double], accumulated: [Double]) throws {
        guard capacities.count == BodySet.count else {
            throw BodySenseError.wrongCount(type: "FatigueModel.capacities", expected: BodySet.count, found: capacities.count)
        }
        guard accumulated.count == BodySet.count else {
            throw BodySenseError.wrongCount(type: "FatigueModel.accumulated", expected: BodySet.count, found: accumulated.count)
        }
        self.capacities = capacities
        self.accumulated = accumulated
    }

    /// One step of `dt` seconds with `effort` per body set (proprio_adapter.py:112-116):
    /// cap += dt (0.05 (1 - cap) - 0.02 effort / 1.0); cap clipped to 0 ... 1; accum += effort
    /// dt. An effort array of other than 10 entries throws.
    public mutating func step(effort: [Double], dt: Double) throws {
        guard effort.count == BodySet.count else {
            throw BodySenseError.wrongCount(type: "FatigueModel.step effort", expected: BodySet.count, found: effort.count)
        }
        for set in 0 ..< BodySet.count {
            let e = effort[set] / FatigueModel.referenceEffort
            let change = dt * (FatigueModel.recoverPerSecond * (1 - capacities[set]) - FatigueModel.drainPerSecond * e)
            capacities[set] = min(max(capacities[set] + change, 0), 1)
            accumulated[set] += effort[set] * dt
        }
    }

    /// Back to fresh: capacities 1, accumulated 0 (proprio_adapter.py:118-120).
    public mutating func reset() {
        self = FatigueModel()
    }
}

// MARK: - The sample of one env step

/// What `ProprioAdapter.read(dt)` returns for one env step (`ProprioReadout`,
/// proprio_adapter.py:123-128), for the 26 rows of `trunk_v1`.
public struct ProprioceptionSample: Equatable, Sendable {
    /// Numbers per bone row: 11 (quaternion x y z w, position cm, velocity cm/s, capacity).
    public static let boneColumns = 11

    /// The simulated time of the env step this sample belongs to, seconds.
    public let time: Double

    /// float32 [26, 11] row-major: per row quaternion x y z w (world; the gaze row relative
    /// to the head), position cm (world), linear velocity cm/s (world), capacity of the row's
    /// body set.
    public let bones: [Float]

    /// float32 [3]: ph_work, pcc_hold, their sum, for this env step (proprio_adapter.py:239-241).
    public let energy: [Float]

    /// float64 [10]: the fatigue capacities after this env step's advance, by `BodySet`
    /// ordinal (the vestibular fields 26-27 read these, vestibular_adapter.py:248).
    public let capacities: [Double]

    /// float64 [10]: the accumulated effort after this env step's advance.
    public let accumulated: [Double]

    /// A sample; counts other than 26 x 11, 3, 10 and 10 throw.
    public init(time: Double, bones: [Float], energy: [Float], capacities: [Double], accumulated: [Double]) throws {
        let boneCount = ProprioceptionLayout.rowCount * ProprioceptionSample.boneColumns
        guard bones.count == boneCount else {
            throw BodySenseError.wrongCount(type: "ProprioceptionSample.bones", expected: boneCount, found: bones.count)
        }
        guard energy.count == 3 else {
            throw BodySenseError.wrongCount(type: "ProprioceptionSample.energy", expected: 3, found: energy.count)
        }
        guard capacities.count == BodySet.count else {
            throw BodySenseError.wrongCount(
                type: "ProprioceptionSample.capacities", expected: BodySet.count, found: capacities.count)
        }
        guard accumulated.count == BodySet.count else {
            throw BodySenseError.wrongCount(
                type: "ProprioceptionSample.accumulated", expected: BodySet.count, found: accumulated.count)
        }
        self.time = time
        self.bones = bones
        self.energy = energy
        self.capacities = capacities
        self.accumulated = accumulated
    }

    /// One bone number by row (0 ... 25) and column (0 ... 10).
    public func bone(row: Int, column: Int) -> Float {
        bones[row * ProprioceptionSample.boneColumns + column]
    }

    /// float32 [10, 2] row-major: (capacity, accumulated) per body set, the `body_sets` array
    /// of proprio_adapter.py:242.
    public var bodySets: [Float] {
        var out: [Float] = []
        out.reserveCapacity(2 * BodySet.count)
        for set in 0 ..< BodySet.count {
            out.append(Float(capacities[set]))
            out.append(Float(accumulated[set]))
        }
        return out
    }
}

// MARK: - The shared tracker

/// The source of the env step's body state for the three body sense readers. `sample(_:)`
/// advances the state when `physics.time` is new and returns the current sample otherwise,
/// so any number of readers may ask within one env step.
public protocol ProprioceptionSampling: AnyObject {
    /// The sample of the env step whose state `physics` holds now.
    func sample(_ physics: any PhysicsReading) throws -> ProprioceptionSample

    /// The fatigue state now.
    var fatigue: FatigueModel { get }

    /// Replaces the fatigue state (a checkpoint's cap and accum) and forgets the last sample.
    func restore(fatigue: FatigueModel)

    /// A fresh fatigue model and no last sample.
    func reset()
}

/// The port of `ProprioAdapter` (proprio_adapter.py:149-246): one fatigue model, advanced
/// once per env step, and the bones, energy and capacities computed from the physics arrays.
/// One instance is shared by `ProprioceptionReader`, `MotorReader` and `VestibularReader`
/// (injected into each at init).
public final class BodyStateTracker: ProprioceptionSampling {
    /// The model indices.
    public let layout: ProprioceptionLayout

    /// The effort formula (`no_gear` in the released run).
    public let effortMode: EffortPowerMode

    /// Seconds per advance: 0.010, one env step (`self.dt`, env.py:3133).
    public let envStepSeconds: Double

    /// The fatigue state after the last advance.
    public private(set) var fatigue: FatigueModel

    private var latest: ProprioceptionSample?

    /// A tracker over `layout`, starting from `fatigue` (fresh by default).
    public init(layout: ProprioceptionLayout,
                effortMode: EffortPowerMode = .noGear,
                envStepSeconds: Double = ModelStep.envStepSeconds,
                fatigue: FatigueModel = FatigueModel()) {
        self.layout = layout
        self.effortMode = effortMode
        self.envStepSeconds = envStepSeconds
        self.fatigue = fatigue
    }

    public func restore(fatigue: FatigueModel) {
        self.fatigue = fatigue
        latest = nil
    }

    public func reset() {
        fatigue = FatigueModel()
        latest = nil
    }

    public func sample(_ physics: any PhysicsReading) throws -> ProprioceptionSample {
        let time = physics.time
        if let latest = latest, latest.time == time {
            return latest
        }
        let next = try advance(physics, time: time)
        latest = next
        return next
    }

    /// Effort per body set (proprio_adapter.py:184-199): |force x qvel[dof]| (times gear under
    /// `withGear`) summed into each actuator's body set.
    public func effortPerSet(actuatorForce: [Double], qvel: [Double]) throws -> [Double] {
        var out = Array(repeating: 0.0, count: BodySet.count)
        for actuator in 0 ..< layout.actuatorBodySets.count {
            guard actuator < actuatorForce.count else {
                throw BodySenseError.arrayTooShort(array: "actuator_force", index: actuator, count: actuatorForce.count)
            }
            let dof = layout.actuatorDofs[actuator]
            guard dof >= 0, dof < qvel.count else {
                throw BodySenseError.arrayTooShort(array: "qvel", index: dof, count: qvel.count)
            }
            // the product in numpy's sequence: tau x gear x qd (with_gear, :191), tau x qd (:193)
            let power: Double
            switch effortMode {
            case .withGear:
                power = actuatorForce[actuator] * layout.actuatorGears[actuator] * qvel[dof]
            case .noGear:
                power = actuatorForce[actuator] * qvel[dof]
            }
            if let set = layout.actuatorBodySets[actuator] {
                out[set.rawValue] += abs(power)
            }
        }
        return out
    }

    private func advance(_ physics: any PhysicsReading, time: Double) throws -> ProprioceptionSample {
        let actuatorForce = physics.actuatorForce
        let qvel = physics.qvel
        let effort = try effortPerSet(actuatorForce: actuatorForce, qvel: qvel)
        // step a copy; `fatigue` takes it only after every read below succeeded
        var stepped = fatigue
        try stepped.step(effort: effort, dt: envStepSeconds)

        let xquat = physics.xquat
        let xpos = physics.xpos
        let columns = ProprioceptionSample.boneColumns
        var bones = Array(repeating: Float(0), count: ProprioceptionLayout.rowCount * columns)
        for (row, body) in layout.bodyIds.enumerated() {
            guard body >= 0, 4 * body + 3 < xquat.count else {
                throw BodySenseError.arrayTooShort(array: "xquat", index: 4 * body + 3, count: xquat.count)
            }
            guard 3 * body + 2 < xpos.count else {
                throw BodySenseError.arrayTooShort(array: "xpos", index: 3 * body + 2, count: xpos.count)
            }
            let velocity = physics.bodyVelocity(bodyId: body)
            guard velocity.count == 6 else {
                throw BodySenseError.wrongCount(type: "bodyVelocity", expected: 6, found: velocity.count)
            }
            let base = row * columns
            // quaternion w x y z -> x y z w (proprio_adapter.py:207-208)
            bones[base + 0] = Float(xquat[4 * body + 1])
            bones[base + 1] = Float(xquat[4 * body + 2])
            bones[base + 2] = Float(xquat[4 * body + 3])
            bones[base + 3] = Float(xquat[4 * body + 0])
            // position cm (proprio_adapter.py:209)
            for axis in 0 ..< 3 {
                bones[base + 4 + axis] = Float(xpos[3 * body + axis] * 100.0)
            }
            // linear velocity cm/s, world frame (proprio_adapter.py:210-211)
            for axis in 0 ..< 3 {
                bones[base + 7 + axis] = Float(velocity[3 + axis] * 100.0)
            }
            // capacity of the row's body set (proprio_adapter.py:212)
            bones[base + 10] = Float(stepped.capacities[BodySet.ofRows[row].rawValue])
        }
        BodyStateTracker.makeGazeRowHeadRelative(&bones)

        // energy (proprio_adapter.py:238-241)
        let effortSum = effort.reduce(0, +)
        let absoluteForceSum = actuatorForce.prefix(layout.actuatorBodySets.count).reduce(0.0) { $0 + abs($1) }
        let work = effortSum * envStepSeconds
        let holding = absoluteForceSum * envStepSeconds * 0.01
        let energy: [Float] = [Float(work), Float(holding), Float(work + holding)]

        let sample = try ProprioceptionSample(
            time: time, bones: bones, energy: energy,
            capacities: stepped.capacities, accumulated: stepped.accumulated)
        fatigue = stepped
        return sample
    }

    /// Replaces the gaze row's quaternion (x y z w) by conj(head) x eye, in float32 as numpy
    /// computes it on the float32 bones (proprio_adapter.py:223-230). An array shorter than
    /// 26 x 11 is left unchanged.
    public static func makeGazeRowHeadRelative(_ bones: inout [Float]) {
        let columns = ProprioceptionSample.boneColumns
        guard bones.count >= ProprioceptionLayout.rowCount * columns else {
            return
        }
        let head = ProprioceptionLayout.headRow * columns
        let gaze = ProprioceptionLayout.gazeRow * columns
        let hx = bones[head], hy = bones[head + 1], hz = bones[head + 2], hw = bones[head + 3]
        let ex = bones[gaze], ey = bones[gaze + 1], ez = bones[gaze + 2], ew = bones[gaze + 3]
        let cx = -hx, cy = -hy, cz = -hz, cw = hw
        let x: Float = cw * ex + cx * ew + cy * ez - cz * ey
        let y: Float = cw * ey - cx * ez + cy * ew + cz * ex
        let z: Float = cw * ez + cx * ey - cy * ex + cz * ew
        let w: Float = cw * ew - cx * ex - cy * ey - cz * ez
        bones[gaze] = x
        bones[gaze + 1] = y
        bones[gaze + 2] = z
        bones[gaze + 3] = w
    }
}

// MARK: - The reader

/// The proprioception `SenseReader` (specification REQ-B-4): float32 [26, 12] per model step,
/// the last env step's bones (columns 0-10, cm and cm/s) with the row index 0 ... 25 in
/// column 11. The tokenizer (WI-30) applies the divisors; see the header.
public final class ProprioceptionReader: SenseReader {
    /// Rows: 26.
    public static let rowCount = ProprioceptionLayout.rowCount

    /// Numbers per row: 12.
    public static let columnCount = 12

    public let sense: SenseName = .proprioception

    private let tracker: ProprioceptionSampling

    /// A reader over the shared tracker.
    public init(tracker: ProprioceptionSampling) {
        self.tracker = tracker
    }

    /// Advances the shared fatigue model once for this env step.
    public func observeEnvStep(_ physics: any PhysicsReading) throws {
        _ = try tracker.sample(physics)
    }

    public func read(_ physics: any PhysicsReading, at step: ModelStep) throws -> SenseReading {
        let sample = try tracker.sample(physics)
        let rows = try ProprioceptionReader.rows(from: sample)
        return SenseReading(sense: .proprioception, primary: rows)
    }

    public func reset() {
        tracker.reset()
    }

    /// The [26, 12] array of one sample: the 11 bone numbers then the row index.
    public static func rows(from sample: ProprioceptionSample) throws -> ShapedArray {
        var values: [Float] = []
        values.reserveCapacity(rowCount * columnCount)
        for row in 0 ..< rowCount {
            for column in 0 ..< ProprioceptionSample.boneColumns {
                values.append(sample.bone(row: row, column: column))
            }
            values.append(Float(row))
        }
        return try ShapedArray(shape: [rowCount, columnCount], values: values)
    }
}
