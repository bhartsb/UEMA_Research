// CommandDecoder.swift: the command (26 handle rows x 9 numbers) into one control per motor.
//
// UEMAKit module UEMACommand, work item WI-37 (target B) of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html, requirement REQ-B-13 and the
// section-5 protocol row "CommandDecoder -> HandleCommandDecoder, injected into Session" of
// docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html. The contract it implements is
// apple/UEMAKit/Specs/command_decoder.md (work item WI-16), which was written from a complete
// read of Source/Python/mimo_bridge/action_decoder.py (527 lines). Every file:line below is
// relative to D:/UEMA_Research/Source/Python unless it starts with another folder. Nothing in
// this file was compiled on the machine that wrote it (Windows 11, no Swift toolchain).
//
// INPUT, per env step (10 ms; 5 env steps per model step of 50 ms, args.json:154 action_repeat 5):
//   - the command: 26 handle rows x 9 numbers, row-major, float64. Columns 0-2 target position
//     in cm (world frame); 3-6 target orientation as a quaternion x y z w (world frame);
//     7 radius (never read: action_decoder.py:43 "Radius is ignored"); 8 strength, clipped to
//     0..1 (action_decoder.py:420). Rows follow HANDLE_NAMES (24, lem/model/body_sets.py:52-75)
//     then lower_trunk, chest (mimo_bridge/body_layout.py:6-14, layout trunk_v1, args.json:140).
//   - the gaze pair (azimuth, elevation) in radians, UEMACore's GazeAngles, or none: when the
//     caller passes none, CommandDecoder below takes UEMACore's `Command.gazeAngles` of row 23
//     (lem_model.py:5122-5140); the saccade generator and the gaze reflex (env.py:2870-2874)
//     may rewrite it before the decoder (command_decoder.md open point A).
//   - the handle mask [26] of Bool (all true on the released configuration, lem_model.py:5723-5724).
//   - the finger goals {MuJoCo joint id: goal angle in rad} for the 44 hand hinges
//     (finger_joint_servo on, args.json:191; hand_target_control.py:457-463), or none.
//   - the body state of this env step, read through BodyStateReading: qpos [nq], qvel [nv],
//     xpos [nbody x 3] m, xmat [nbody x 9], xaxis [njnt x 3], xanchor [njnt x 3] m.
// OPERATION: action_decoder.py decode (lines 402-520), ported line for line in
//   HandleCommandDecoder.decode: per handle chain a damped least squares of the rotation still
//   needed (and, for a position handle, the wanted move weighted 8.0 rad per metre) over the
//   chain's hinge axes; per hinge u = strength x (6.0 x dq - 0.3 x hinge speed); the three
//   handles of one arm solved together; a direction-preserving rescale per chain and per arm
//   group; a clip to the control bounds; the eye servo u = 80.0 x (target - angle) - 3.0 x speed.
// OUTPUT: ctrl, one Double per motor, each within its control bounds (-1..1 narrowed to the
//   actuator force range as float32, actuation.py:136-138), to be written into data.ctrl through
//   UEMACore's `Physics.write(controls:)` (MotorCommandWriter below; actuation.py:153-154).
//   THE MOTOR COUNT. UEMACore's MotorControls holds exactly 90 numbers, the nu of the exported
//   crib scene (Sources/UEMAScene/Resources/crib_age24_trunk_v1/bundle.json "nu": 90; env.py:2918
//   "18 of 90 actuators"; MuJoCoPhysics.swift header "THE MOTOR COUNT"); it read 92 before
//   November 4, 2026. CommandDecoder below refuses an engine of any other count with
//   CommandDecoderError.inconsistentTables. HandleCommandDecoder itself works for any nu.
//
// SOURCES READ ENTIRE BEFORE WRITING (November 4, 2026): action_decoder.py (527 lines);
// apple/UEMAKit/Specs/command_decoder.md; Specs/proprio_rows.md; Specs/motor_controller_input.md;
// mimo_bridge/body_layout.py (83 lines); mimo_bridge/tests/test_command_moves_body.py (138 lines);
// the cited lines of env.py (985-1000, 1288-1310, 2832-2891), proprio_adapter.py (1-70, 160-200),
// lem/model/body_sets.py (1-105), gaze_reflex.py (30-50), lem_model.py (5120-5145),
// hand_target_control.py (455-465), third_party/MIMo/mimo/mimo_infant/actuation/actuation.py
// (100-175), MIMo_metav2.xml (225-345: the 90 motors of the base body) and the released
// args.json (external/D/UEMA_preserved/b1_continue_B1K_skip13_492800/run/args.json: lines 19, 29,
// 30, 31, 51, 80, 113, 140, 154, 187-191, 215).
//
// MUJOCO FIELDS THE ADAPTER READS (this file names no MuJoCo C symbol; see "WHAT IS NOT HERE").
// Field names, element types and sizes from https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html
// (sections "mjModel" and "mjData", read November 4, 2026) and from the MuJoCo 3.3.0 headers
// shipped in the mujoco 3.3.0 Python wheel (mjmodel.h, mjdata.h; mujoco.h line 19
// "#define mjVERSION_HEADER 330"), read the same day:
//   mjModel: "mjtSize nu; number of scalar controls = dim(ctrl)"; "int* jnt_type; type of joint
//   (mjtJoint) (njnt x 1)"; "int* jnt_bodyid; id of joint's body (njnt x 1)"; "int* jnt_qposadr;
//   start addr in 'qpos' for joint's data (njnt x 1)"; "int* jnt_dofadr; start addr in 'qvel'
//   for joint's data (njnt x 1)"; "mjtByte* jnt_limited; does joint have limits (njnt x 1)";
//   "mjtNum* jnt_range; joint limits (njnt x 2)"; "int* body_parentid; id of body's parent
//   (nbody x 1)"; "int* actuator_trntype; transmission type (mjtTrn) (nu x 1)";
//   "int* actuator_trnid; transmission id: joint, tendon, site (nu x 2)"; "mjtByte*
//   actuator_ctrllimited; is control limited (nu x 1)"; "mjtNum* actuator_ctrlrange; range of
//   controls (nu x 2)"; "mjtNum* actuator_forcerange; range of forces (nu x 2)"; "mjtNum* qpos0;
//   qpos values at default pose (nq x 1)".
//   mjData: "mjtNum* qpos; position (nq x 1)"; "mjtNum* qvel; velocity (nv x 1)"; "mjtNum* ctrl;
//   control (nu x 1)"; "mjtNum* xpos; Cartesian position of body frame (nbody x 3)"; "mjtNum*
//   xmat; Cartesian orientation of body frame (nbody x 9)"; "mjtNum* xaxis; Cartesian joint axis
//   (njnt x 3)"; "mjtNum* xanchor; Cartesian position of joint anchor (njnt x 3)".
//   Enum values (mjmodel.h lines 87-90, 203, 225, 233): mjJNT_FREE = 0, mjJNT_BALL = 1,
//   mjJNT_SLIDE = 2, mjJNT_HINGE = 3; mjTRN_JOINT = 0; mjGAIN_FIXED = 0; mjBIAS_NONE = 0.
//   Functions the reach measurement needs (https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html,
//   read November 4, 2026): "mjData* mj_makeData(const mjModel* m); Allocate mjData corresponding
//   to given model."; "void mj_forward(const mjModel* m, mjData* d); Forward dynamics: same as
//   mj_step but do not integrate in time."; "void mj_deleteData(mjData* d); Free memory
//   allocation in mjData."; "int mj_name2id(const mjModel* m, int type, const char* name); Get id
//   of object with the specified mjtObj type and name, returns -1 if id not found."
//
// THE TYPES, ONE RESPONSIBILITY EACH.
//   HandleCommandDecoder (ControlDecoding): the arithmetic of action_decoder.py over [Double]
//     arrays and an injected BodyStateReading.
//   PhysicsBodyState (BodyStateReading): UEMACore's PhysicsReading flat arrays as the
//     decoder's per-body and per-joint reads.
//   CommandDecoder (UEMACore's CommandDecoding): UEMACore's Command, HandleMask, GazeAngles
//     and FingerJointGoals into the engine's arrays, over the physics given at init; returns
//     UEMACore's MotorControls.
//   MotorCommandWriter: one env step's decode, the write through UEMACore's Physics, and the
//     body-function modulation (MotorFunction.swift) in the Python order (env.py:2884-2893).
//   ActuatedBodyTables(modelTables:), ScratchForwardKinematics, CommandDecoder.released(physics:):
//     the adapters over UEMAPhysics (WI-25) described below.
//
// THE ADAPTERS OVER UEMAPhysics (end of this file). The decoder takes the time-invariant model
// constants as the value type ActuatedBodyTables and the reach measurement as
// ForwardKinematicsQuerying, so its tests need no MuJoCo. UEMAPhysics (WI-25,
// Sources/UEMAPhysics/PhysicsState.swift, on disk November 4, 2026) gives the same numbers in
// plain arrays: PhysicsModelTables through PhysicsModelDescribing.modelTables(), the scratch
// forward kinematics through ScratchKinematicsQuerying.scratchBodyPosition, and the two model
// writes of the modulation through ActuatorParameterWriting. ActuatedBodyTables(modelTables:),
// ScratchForwardKinematics, ActuatorParameterStrengthWriter and CommandDecoder.released(physics:)
// map them one to one. This file names no MuJoCo C symbol.
//
// FIRST BUILD ON A MAC (work item WI-37's acceptance; plan section 8 steps 1 to 4 first):
//  1. `cd apple/UEMAKit && xcodebuild -scheme UEMAKit-Package build -destination 'platform=macOS'`
//     exits 0 with UEMACommand compiled. A compiler error in this file or MotorFunction.swift is
//     copied into apple/records/first_build.md (REQ-B-18 item 4) and fixed here.
//  2. `xcodebuild -scheme UEMAKit-Package test -destination 'platform=macOS' -only-testing:UEMACommandTests`
//     prints every test of Tests/UEMACommandTests passing: 31 in all (HandleCommandDecoderTests 14,
//     CommandDecoderTests 10, MotorFunctionTests 6, UEMACommandPackageTests 1). The acceptance
//     pattern on the test plant is CommandDecoderTests.testHeldCommandMovesTheBodyPartInTheAskedDirection.
//  3. On the loaded WI-15 bundle: `CommandDecoder.released(physics: MuJoCoPhysics)` builds the
//     tables, the reach and the decoder (end of this file); the body-function side is
//     `MotorFunctionApplier(modulator: MotorFunctionModulator(tables: ActuatedBodyTables(
//     modelTables: physics.modelTables())), mode: .modulate, capacities:
//     FatigueCapacities(source: <the shared UEMASenses BodyStateTracker>), writer:
//     ActuatorParameterStrengthWriter(physics))`, passed to `MotorCommandWriter(decoder:physics:
//     motorFunction:)`. Check: tables.actuatorCount (90 in bundle.json; see THE MOTOR COUNT
//     DISAGREES above, which makes `released` throw until UEMACore says 90); CornerSweepReach
//     gives, in cm, wrists 42.85, thumbs 5.54, index 5.48, middle 6.09, ring 5.48, pinky 6.80,
//     head 4.81, lower_trunk 5.58 (model_receipt.json motor_synergy.pos_reach_cm, read November
//     4, 2026) within 0.01 cm.
//  4. The acceptance of this row (test_command_moves_body.py's pattern): on the loaded bundle,
//     hold one command for 25 env steps per handle in upperarm_R, upperarm_L, head, thigh_R
//     (full strength, the body's current orientation turned 40 degrees about x) and check the
//     largest |qpos change| over the body exceeds 1e-3 rad; a zero-strength copy moves less than
//     the full-strength one; every control of every env step lies within -1..1.
//  5. Fixture equality (REQ-B-13, T-B-3): the 200 recorded ctrl[92] vectors of the fixture
//     against this decoder's, within 1e-6, with the gaze pair recorded AS PASSED to decode
//     (command_decoder.md open point A).
//  6. Record each command, exit code and clock time (AM/PM Pacific) in apple/FIRST_BUILD_RECORD.md (WI-41).
//  7. Constructs to check first if step 1 fails: the existential parameters (`BodyStateReading`,
//     `ForwardKinematicsQuerying`, `ReachMeasuring` written without `any`, accepted in Swift 5
//     language mode); throwing before every stored property is set in the designated init of
//     the final class HandleCommandDecoder; `public private(set) var` on a class property; the
//     generic `released<P>` whose constraint joins three class-bound protocols of two modules;
//     the names taken from UEMAPhysics (PhysicsModelTables, PhysicsModelDescribing,
//     ScratchKinematicsQuerying, ActuatorParameterWriting): if WI-25 renamed one, follow the
//     landed name here.

import Foundation
import UEMACore
import UEMAPhysics

// MARK: - Small vector and matrix arithmetic (no framework dependency)

/// A 3-vector of Double: a position in metres, a rotation vector in radians or a joint axis.
public struct Vector3: Equatable, Sendable {
    /// The x component.
    public var x: Double
    /// The y component.
    public var y: Double
    /// The z component.
    public var z: Double

    /// Makes the vector from its three components.
    public init(_ x: Double, _ y: Double, _ z: Double) {
        self.x = x
        self.y = y
        self.z = z
    }

    /// Makes the vector from an array of exactly three numbers; any other count gives zero.
    public init(array: [Double]) {
        if array.count == 3 {
            self.init(array[0], array[1], array[2])
        } else {
            self.init(0, 0, 0)
        }
    }

    /// The zero vector.
    public static let zero = Vector3(0, 0, 0)

    /// The three components as an array [x, y, z].
    public var array: [Double] { [x, y, z] }

    /// Euclidean length.
    public var norm: Double { (x * x + y * y + z * z).squareRoot() }

    /// Component-wise sum.
    public static func + (a: Vector3, b: Vector3) -> Vector3 { Vector3(a.x + b.x, a.y + b.y, a.z + b.z) }

    /// Component-wise difference.
    public static func - (a: Vector3, b: Vector3) -> Vector3 { Vector3(a.x - b.x, a.y - b.y, a.z - b.z) }

    /// Scalar product.
    public static func * (s: Double, v: Vector3) -> Vector3 { Vector3(s * v.x, s * v.y, s * v.z) }

    /// Negation.
    public static prefix func - (v: Vector3) -> Vector3 { Vector3(-v.x, -v.y, -v.z) }

    /// Dot product.
    public func dot(_ other: Vector3) -> Double { x * other.x + y * other.y + z * other.z }

    /// Cross product a x b, the same component formula numpy.cross uses
    /// (action_decoder.py:385, :471 call np.cross on the axis and the lever arm).
    public func cross(_ b: Vector3) -> Vector3 {
        Vector3(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x)
    }

    /// Each component clipped to -limit..limit (action_decoder.py:382 `np.clip(e, -reach, reach)`).
    public func clipped(to limit: Double) -> Vector3 {
        Vector3(min(max(x, -limit), limit), min(max(y, -limit), limit), min(max(z, -limit), limit))
    }

    /// True when every component is finite (action_decoder.py:377 `np.all(np.isfinite(...))`).
    public var isFinite: Bool { x.isFinite && y.isFinite && z.isFinite }
}

/// A 3 x 3 rotation matrix stored row-major, the layout of MuJoCo's xmat (9 numbers per body).
public struct Matrix3: Equatable, Sendable {
    /// The nine entries, row-major: m[3 * row + column].
    public var m: [Double]

    /// Makes the matrix from nine row-major numbers; any other count gives the identity.
    public init(rowMajor: [Double]) {
        if rowMajor.count == 9 {
            m = rowMajor
        } else {
            m = Matrix3.identity.m
        }
    }

    /// The identity rotation.
    public static let identity = Matrix3(rowMajor: [1, 0, 0, 0, 1, 0, 0, 0, 1])

    /// Entry at (row, column).
    public subscript(row: Int, column: Int) -> Double {
        get { m[3 * row + column] }
        set { m[3 * row + column] = newValue }
    }

    /// The transpose.
    public var transposed: Matrix3 {
        Matrix3(rowMajor: [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]])
    }

    /// The trace (sum of the diagonal).
    public var trace: Double { m[0] + m[4] + m[8] }

    /// Matrix product a b.
    public static func * (a: Matrix3, b: Matrix3) -> Matrix3 {
        var out = [Double](repeating: 0, count: 9)
        for r in 0..<3 {
            for c in 0..<3 {
                var s = 0.0
                for k in 0..<3 { s += a[r, k] * b[k, c] }
                out[3 * r + c] = s
            }
        }
        return Matrix3(rowMajor: out)
    }

    /// The rotation matrix of a quaternion given as x y z w, normalised first; a quaternion of
    /// norm under 1e-12 gives the identity. Port of `quat_xyzw_to_mat` (action_decoder.py:150-160).
    public init(quaternionXYZW q: [Double]) {
        guard q.count == 4 else {
            self = Matrix3.identity
            return
        }
        var x = q[0], y = q[1], z = q[2], w = q[3]
        let n = (x * x + y * y + z * z + w * w).squareRoot()
        if n < 1e-12 {
            self = Matrix3.identity
            return
        }
        x /= n
        y /= n
        z /= n
        w /= n
        m = [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ]
    }

    /// The rotation vector (axis x angle, angle in 0..pi) of this rotation matrix. Port of
    /// `rotmat_log` (action_decoder.py:167-187): angle = arccos(clamp((trace - 1) / 2, -1, 1));
    /// under 1e-9 the zero vector; within 1e-6 of pi the strongest column of
    /// (R + R^T + 2 I) / 4 with the sign of the skew part; otherwise the skew part
    /// (R21 - R12, R02 - R20, R10 - R01) x angle / (2 sin angle).
    public func logarithm() -> Vector3 {
        var c = (trace - 1.0) / 2.0
        c = min(1.0, max(-1.0, c))
        let angle = acos(c)
        if angle < 1e-9 {
            return Vector3.zero
        }
        let skew = Vector3(self[2, 1] - self[1, 2], self[0, 2] - self[2, 0], self[1, 0] - self[0, 1])
        if abs(Double.pi - angle) < 1e-6 {
            let t = transposed
            var a = [Double](repeating: 0, count: 9)
            for i in 0..<9 { a[i] = (m[i] + t.m[i]) / 4.0 }
            a[0] += 0.5
            a[4] += 0.5
            a[8] += 0.5
            var best = 0
            if a[4] > a[0] { best = 1 }
            if a[8] > a[3 * best + best] { best = 2 }
            var axis = Vector3(a[best], a[3 + best], a[6 + best])
            if axis.dot(skew) < 0.0 {
                axis = -axis
            }
            let n = axis.norm
            return n > 0 ? (angle / n) * axis : Vector3.zero
        }
        return (angle / (2.0 * sin(angle))) * skew
    }
}

/// Solves min |A dq - w|^2 + damping |dq|^2 for dq, with A given as rows of n numbers.
/// Port of `damped_least_squares` (action_decoder.py:190-196): dq = (A^T A + damping I)^-1 A^T w,
/// the n x n system solved by Gaussian elimination with partial pivoting (numpy.linalg.solve's
/// LU with pivoting gives the same solution of a well-posed system).
public enum DampedLeastSquares {
    /// Returns dq of length `columns`; an empty system (0 columns) returns an empty array.
    /// Throws CommandDecoderError.singularSystem when a pivot is 0 (impossible for damping > 0
    /// unless an entry is not finite).
    public static func solve(rows: [[Double]], rhs: [Double], columns n: Int, damping: Double) throws -> [Double] {
        if n == 0 {
            return []
        }
        var ata = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
        var atw = [Double](repeating: 0, count: n)
        for (r, row) in rows.enumerated() {
            let w = r < rhs.count ? rhs[r] : 0.0
            for i in 0..<n {
                let ri = i < row.count ? row[i] : 0.0
                atw[i] += ri * w
                for j in 0..<n {
                    let rj = j < row.count ? row[j] : 0.0
                    ata[i][j] += ri * rj
                }
            }
        }
        for i in 0..<n { ata[i][i] += damping }
        return try LinearSystem.solve(matrix: ata, rhs: atw)
    }
}

/// A dense square linear solve, Gaussian elimination with partial pivoting.
public enum LinearSystem {
    /// Solves matrix x = rhs for x. Throws CommandDecoderError.singularSystem when the largest
    /// available pivot is 0 or not finite.
    public static func solve(matrix: [[Double]], rhs: [Double]) throws -> [Double] {
        let n = rhs.count
        if n == 0 {
            return []
        }
        var a = matrix
        var b = rhs
        for col in 0..<n {
            var pivotRow = col
            var pivotMagnitude = abs(a[col][col])
            for r in (col + 1)..<max(col + 1, n) where abs(a[r][col]) > pivotMagnitude {
                pivotMagnitude = abs(a[r][col])
                pivotRow = r
            }
            if !(pivotMagnitude > 0) || !pivotMagnitude.isFinite {
                throw CommandDecoderError.singularSystem
            }
            if pivotRow != col {
                a.swapAt(pivotRow, col)
                b.swapAt(pivotRow, col)
            }
            let pivot = a[col][col]
            for r in (col + 1)..<max(col + 1, n) {
                let factor = a[r][col] / pivot
                if factor == 0 { continue }
                for c in col..<n { a[r][c] -= factor * a[col][c] }
                b[r] -= factor * b[col]
            }
        }
        var x = [Double](repeating: 0, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            var s = b[i]
            for c in (i + 1)..<max(i + 1, n) { s -= a[i][c] * x[c] }
            x[i] = s / a[i][i]
        }
        return x
    }
}

// MARK: - Errors

/// Everything the decoder can refuse, each naming the fact that failed.
public enum CommandDecoderError: Error, Equatable, Sendable {
    /// Two arrays of the body tables disagree in length (the message names them).
    case inconsistentTables(String)
    /// A body or joint the handle layout names is not in the tables.
    case missingBody(String)
    /// The command has the wrong count of numbers (expected, found).
    case commandCount(expected: Int, found: Int)
    /// The handle mask has the wrong count (expected, found).
    case maskCount(expected: Int, found: Int)
    /// A finger hinge has no entry in the finger goals (joint ids).
    case missingFingerGoal([Int])
    /// The state's qpos or qvel is shorter than an address the tables use.
    case stateTooShort(String)
    /// The damped least squares met a zero or non-finite pivot.
    case singularSystem
}

// MARK: - The body tables and the state the decoder reads

/// The constant tables of one actuated body, read once from mjModel (field list in this file's
/// header). A value type: the adapter in UEMAPhysics fills it; tests fill it by hand.
public struct ActuatedBodyTables: Equatable, Sendable {
    /// mjtJoint value of a hinge (mjmodel.h:87-90: mjJNT_FREE 0, BALL 1, SLIDE 2, HINGE 3).
    public static let hingeJointType = 3
    /// mjtTrn value of a joint transmission (mjmodel.h:203 mjTRN_JOINT = 0).
    public static let jointTransmission = 0
    /// mjtGain value of a fixed gain (mjmodel.h:225 mjGAIN_FIXED = 0).
    public static let fixedGainType = 0
    /// mjtBias value of no bias (mjmodel.h:233 mjBIAS_NONE = 0).
    public static let noBiasType = 0

    /// nu, the number of motors (90 on the released body).
    public let actuatorCount: Int
    /// Body names by body id (mj_id2name with mjOBJ_BODY), body 0 the world.
    public let bodyNames: [String]
    /// body_parentid [nbody].
    public let bodyParentID: [Int]
    /// Joint names by joint id (mj_id2name with mjOBJ_JOINT).
    public let jointNames: [String]
    /// jnt_type [njnt] (mjtJoint).
    public let jointType: [Int]
    /// jnt_bodyid [njnt].
    public let jointBodyID: [Int]
    /// jnt_qposadr [njnt].
    public let jointQposAddress: [Int]
    /// jnt_dofadr [njnt].
    public let jointDofAddress: [Int]
    /// jnt_limited [njnt].
    public let jointLimited: [Bool]
    /// jnt_range [njnt], column 0.
    public let jointRangeLow: [Double]
    /// jnt_range [njnt], column 1.
    public let jointRangeHigh: [Double]
    /// actuator_trntype [nu] (mjtTrn).
    public let actuatorTransmissionType: [Int]
    /// actuator_trnid [nu], column 0 (the joint id under mjTRN_JOINT).
    public let actuatorTransmissionID: [Int]
    /// actuator_ctrllimited [nu].
    public let actuatorControlLimited: [Bool]
    /// actuator_ctrlrange [nu], column 0.
    public let actuatorControlRangeLow: [Double]
    /// actuator_ctrlrange [nu], column 1.
    public let actuatorControlRangeHigh: [Double]
    /// actuator_forcerange [nu], column 0 (MIMo's action space low, actuation.py:136-138).
    public let actuatorForceRangeLow: [Double]
    /// actuator_forcerange [nu], column 1.
    public let actuatorForceRangeHigh: [Double]
    /// actuator_gaintype [nu] (mjtGain).
    public let actuatorGainType: [Int]
    /// actuator_biastype [nu] (mjtBias).
    public let actuatorBiasType: [Int]
    /// actuator_gainprm [nu], column 0 of mjNGAIN = 10 (mjmodel.h:39): the fixed force gain.
    public let actuatorGainParameter0: [Double]

    /// Makes the tables, checking that every per-joint array has jointNames.count entries,
    /// every per-actuator array has actuatorCount entries and bodyParentID has bodyNames.count.
    public init(actuatorCount: Int,
                bodyNames: [String],
                bodyParentID: [Int],
                jointNames: [String],
                jointType: [Int],
                jointBodyID: [Int],
                jointQposAddress: [Int],
                jointDofAddress: [Int],
                jointLimited: [Bool],
                jointRangeLow: [Double],
                jointRangeHigh: [Double],
                actuatorTransmissionType: [Int],
                actuatorTransmissionID: [Int],
                actuatorControlLimited: [Bool],
                actuatorControlRangeLow: [Double],
                actuatorControlRangeHigh: [Double],
                actuatorForceRangeLow: [Double],
                actuatorForceRangeHigh: [Double],
                actuatorGainType: [Int],
                actuatorBiasType: [Int],
                actuatorGainParameter0: [Double]) throws {
        let njnt = jointNames.count
        let perJoint: [(String, Int)] = [
            ("jointType", jointType.count), ("jointBodyID", jointBodyID.count),
            ("jointQposAddress", jointQposAddress.count), ("jointDofAddress", jointDofAddress.count),
            ("jointLimited", jointLimited.count), ("jointRangeLow", jointRangeLow.count),
            ("jointRangeHigh", jointRangeHigh.count),
        ]
        for (name, count) in perJoint where count != njnt {
            throw CommandDecoderError.inconsistentTables("\(name) has \(count) entries, jointNames \(njnt)")
        }
        let perActuator: [(String, Int)] = [
            ("actuatorTransmissionType", actuatorTransmissionType.count),
            ("actuatorTransmissionID", actuatorTransmissionID.count),
            ("actuatorControlLimited", actuatorControlLimited.count),
            ("actuatorControlRangeLow", actuatorControlRangeLow.count),
            ("actuatorControlRangeHigh", actuatorControlRangeHigh.count),
            ("actuatorForceRangeLow", actuatorForceRangeLow.count),
            ("actuatorForceRangeHigh", actuatorForceRangeHigh.count),
            ("actuatorGainType", actuatorGainType.count), ("actuatorBiasType", actuatorBiasType.count),
            ("actuatorGainParameter0", actuatorGainParameter0.count),
        ]
        for (name, count) in perActuator where count != actuatorCount {
            throw CommandDecoderError.inconsistentTables("\(name) has \(count) entries, actuatorCount \(actuatorCount)")
        }
        if bodyParentID.count != bodyNames.count {
            throw CommandDecoderError.inconsistentTables("bodyParentID has \(bodyParentID.count) entries, bodyNames \(bodyNames.count)")
        }
        for (j, b) in jointBodyID.enumerated() where b < 0 || b >= bodyNames.count {
            throw CommandDecoderError.inconsistentTables("joint \(j) names body \(b) outside 0..<\(bodyNames.count)")
        }
        // The arm groups walk body_parentid up to the world (action_decoder.py:337-341) and the
        // decoder indexes qpos and qvel by these addresses, so each must be a valid index.
        for (b, parent) in bodyParentID.enumerated() where parent < 0 || parent >= bodyNames.count {
            throw CommandDecoderError.inconsistentTables("body \(b) names parent \(parent) outside 0..<\(bodyNames.count)")
        }
        for (j, address) in jointQposAddress.enumerated() where address < 0 {
            throw CommandDecoderError.inconsistentTables("joint \(j) has qpos address \(address)")
        }
        for (j, address) in jointDofAddress.enumerated() where address < 0 {
            throw CommandDecoderError.inconsistentTables("joint \(j) has qvel address \(address)")
        }
        for (a, j) in actuatorTransmissionID.enumerated()
        where actuatorTransmissionType[a] == ActuatedBodyTables.jointTransmission && (j < 0 || j >= njnt) {
            throw CommandDecoderError.inconsistentTables("actuator \(a) names joint \(j) outside 0..<\(njnt)")
        }
        self.actuatorCount = actuatorCount
        self.bodyNames = bodyNames
        self.bodyParentID = bodyParentID
        self.jointNames = jointNames
        self.jointType = jointType
        self.jointBodyID = jointBodyID
        self.jointQposAddress = jointQposAddress
        self.jointDofAddress = jointDofAddress
        self.jointLimited = jointLimited
        self.jointRangeLow = jointRangeLow
        self.jointRangeHigh = jointRangeHigh
        self.actuatorTransmissionType = actuatorTransmissionType
        self.actuatorTransmissionID = actuatorTransmissionID
        self.actuatorControlLimited = actuatorControlLimited
        self.actuatorControlRangeLow = actuatorControlRangeLow
        self.actuatorControlRangeHigh = actuatorControlRangeHigh
        self.actuatorForceRangeLow = actuatorForceRangeLow
        self.actuatorForceRangeHigh = actuatorForceRangeHigh
        self.actuatorGainType = actuatorGainType
        self.actuatorBiasType = actuatorBiasType
        self.actuatorGainParameter0 = actuatorGainParameter0
    }

    /// njnt.
    public var jointCount: Int { jointNames.count }
    /// nbody.
    public var bodyCount: Int { bodyNames.count }

    /// The id of the body with this name, or nil (the adapter's mj_name2id returns -1).
    public func bodyID(named name: String) -> Int? {
        bodyNames.firstIndex(of: name)
    }

    /// The id of the joint with this name, or nil.
    public func jointID(named name: String) -> Int? {
        jointNames.firstIndex(of: name)
    }
}

/// The body's state at one env step, as the decoder reads it from mjData (field list in this
/// file's header). The adapter in UEMAPhysics conforms; tests use a fixed value.
public protocol BodyStateReading {
    /// data.qpos, nq numbers (joint positions; a hinge's angle in rad at jnt_qposadr).
    var qpos: [Double] { get }
    /// data.qvel, nv numbers (joint velocities; a hinge's rad/s at jnt_dofadr).
    var qvel: [Double] { get }
    /// data.xpos[body], metres, world frame.
    func bodyPosition(_ body: Int) -> Vector3
    /// data.xmat[body], the body's rotation, row-major.
    func bodyRotation(_ body: Int) -> Matrix3
    /// data.xaxis[joint], the hinge axis in the world frame (unit vector).
    func jointAxis(_ joint: Int) -> Vector3
    /// data.xanchor[joint], the hinge anchor in the world frame, metres.
    func jointAnchor(_ joint: Int) -> Vector3
}

/// A forward-kinematics query on a scratch copy of the body, for the reach measurement.
/// The adapter in UEMAPhysics implements it as: a fresh mjData (mj_makeData: qpos at qpos0),
/// each listed joint's qpos entry set to its angle, mj_forward, read xpos[body], mj_deleteData;
/// nothing the caller holds is disturbed (action_decoder.py:216-217, 224-238).
public protocol ForwardKinematicsQuerying {
    /// The world position (metres) of `body` when every joint in `jointAngles` holds the given
    /// angle (rad) and every other joint holds its default-pose value.
    func bodyPosition(ofBody body: Int, withJointAngles jointAngles: [Int: Double]) throws -> Vector3
}

// MARK: - The handle layout

/// The 26 handle names, the bodies they stand on, the chains of bodies whose hinges each may
/// move, and the handles whose position row enters the solve. `trunkV1` is the released body's.
public struct HandleLayout: Equatable, Sendable {
    /// Handle names in row sequence of the command.
    public let names: [String]
    /// Handle name -> MuJoCo body name of the handle's own bone.
    public let bones: [String: String]
    /// Handle name -> body names whose hinges the handle may move, distal first; empty for gaze.
    public let chains: [String: [String]]
    /// Handles whose position row (columns 0-2) enters the solve.
    public let positionHandles: Set<String>
    /// The first words of the finger handle names (FINGER_DIGITS, action_decoder.py:72).
    public let fingerDigits: Set<String>
    /// The eye joints the gaze servo drives: joint name, gaze component (0 azimuth, 1 elevation)
    /// and sign (EYE_JOINTS, action_decoder.py:141-146; the right eye's horizontal axis is
    /// 0 0 -1 in MIMo_modelv2.xml).
    public let eyeJoints: [EyeJoint]

    /// One entry of EYE_JOINTS.
    public struct EyeJoint: Equatable, Sendable {
        /// The MuJoCo joint name, for example "robot:left_eye_horizontal".
        public let jointName: String
        /// 0 for azimuth, 1 for elevation.
        public let gazeComponent: Int
        /// +1.0 or -1.0.
        public let sign: Double

        /// Makes one entry.
        public init(jointName: String, gazeComponent: Int, sign: Double) {
            self.jointName = jointName
            self.gazeComponent = gazeComponent
            self.sign = sign
        }
    }

    /// Makes a layout; tests build small ones, production uses `trunkV1`.
    public init(names: [String],
                bones: [String: String],
                chains: [String: [String]],
                positionHandles: Set<String>,
                fingerDigits: Set<String>,
                eyeJoints: [EyeJoint]) {
        self.names = names
        self.bones = bones
        self.chains = chains
        self.positionHandles = positionHandles
        self.fingerDigits = fingerDigits
        self.eyeJoints = eyeJoints
    }

    /// HANDLE_NAMES (lem/model/body_sets.py:52-75): 24 names, index 23 the gaze handle.
    public static let legacyNames: [String] = [
        "wrist_L", "thumb_L", "index_L", "middle_L", "ring_L", "pinky_L",
        "wrist_R", "thumb_R", "index_R", "middle_R", "ring_R", "pinky_R",
        "head",
        "upperarm_L", "lowerarm_L",
        "upperarm_R", "lowerarm_R",
        "thigh_L", "calf_L",
        "thigh_R", "calf_R",
        "foot_L", "foot_R",
        "gaze",
    ]

    /// TRUNK_NAMES (body_layout.py:6): the two rows layout trunk_v1 adds after the gaze row.
    public static let trunkNames: [String] = ["lower_trunk", "chest"]

    /// The released body's layout: names_for('trunk_v1') (body_layout.py:10-15), BONE_MAP
    /// (proprio_adapter.py:27-41) plus TRUNK_BODIES (body_layout.py:7), HANDLE_CHAINS
    /// (action_decoder.py:75-99) plus the trunk chains (action_decoder.py:267), POSITION_HANDLES
    /// with position_rows 'all' (action_decoder.py:118-120, :132-136; args.json:190).
    public static let trunkV1 = HandleLayout(
        names: legacyNames + trunkNames,
        bones: [
            "wrist_L": "left_hand", "thumb_L": "left_thdistal", "index_L": "left_ffdistal",
            "middle_L": "left_mfdistal", "ring_L": "left_rfdistal", "pinky_L": "left_lfdistal",
            "wrist_R": "right_hand", "thumb_R": "right_thdistal", "index_R": "right_ffdistal",
            "middle_R": "right_mfdistal", "ring_R": "right_rfdistal", "pinky_R": "right_lfdistal",
            "head": "head",
            "upperarm_L": "left_upper_arm", "lowerarm_L": "left_lower_arm",
            "upperarm_R": "right_upper_arm", "lowerarm_R": "right_lower_arm",
            "thigh_L": "left_upper_leg", "calf_L": "left_lower_leg",
            "thigh_R": "right_upper_leg", "calf_R": "right_lower_leg",
            "foot_L": "left_foot", "foot_R": "right_foot",
            "gaze": "left_eye",
            "lower_trunk": "lower_body", "chest": "chest",
        ],
        chains: [
            "wrist_L": ["left_hand", "left_lower_arm", "left_upper_arm"],
            "wrist_R": ["right_hand", "right_lower_arm", "right_upper_arm"],
            "thumb_L": ["left_thdistal", "left_thhub", "left_thbase"],
            "thumb_R": ["right_thdistal", "right_thhub", "right_thbase"],
            "index_L": ["left_ffdistal", "left_ffmiddle", "left_ffknuckle"],
            "index_R": ["right_ffdistal", "right_ffmiddle", "right_ffknuckle"],
            "middle_L": ["left_mfdistal", "left_mfmiddle", "left_mfknuckle"],
            "middle_R": ["right_mfdistal", "right_mfmiddle", "right_mfknuckle"],
            "ring_L": ["left_rfdistal", "left_rfmiddle", "left_rfknuckle"],
            "ring_R": ["right_rfdistal", "right_rfmiddle", "right_rfknuckle"],
            "pinky_L": ["left_lfdistal", "left_lfmiddle", "left_lfknuckle", "left_lfmetacarpal"],
            "pinky_R": ["right_lfdistal", "right_lfmiddle", "right_lfknuckle", "right_lfmetacarpal"],
            "head": ["head"],
            "upperarm_L": ["left_upper_arm"], "lowerarm_L": ["left_lower_arm"],
            "upperarm_R": ["right_upper_arm"], "lowerarm_R": ["right_lower_arm"],
            "thigh_L": ["left_upper_leg"], "calf_L": ["left_lower_leg"],
            "thigh_R": ["right_upper_leg"], "calf_R": ["right_lower_leg"],
            "foot_L": ["left_foot"], "foot_R": ["right_foot"],
            // Empty on purpose: the eye hinges belong to the gaze servo (action_decoder.py:94-98).
            "gaze": [],
            "lower_trunk": ["lower_body"], "chest": ["chest"],
        ],
        positionHandles: [
            "wrist_L", "wrist_R", "thumb_L", "index_L", "middle_L", "ring_L", "pinky_L",
            "thumb_R", "index_R", "middle_R", "ring_R", "pinky_R",
            "head", "lower_trunk",
        ],
        fingerDigits: ["thumb", "index", "middle", "ring", "pinky"],
        eyeJoints: [
            EyeJoint(jointName: "robot:left_eye_horizontal", gazeComponent: 0, sign: 1.0),
            EyeJoint(jointName: "robot:left_eye_vertical", gazeComponent: 1, sign: 1.0),
            EyeJoint(jointName: "robot:right_eye_horizontal", gazeComponent: 0, sign: -1.0),
            EyeJoint(jointName: "robot:right_eye_vertical", gazeComponent: 1, sign: 1.0),
        ]
    )

    /// The row index of a handle name, or nil.
    public func index(of name: String) -> Int? {
        names.firstIndex(of: name)
    }

    /// True when the handle's first word (before "_") is a finger digit (action_decoder.py:356-358).
    public func isFingerHandle(_ name: String) -> Bool {
        let first = name.split(separator: "_", maxSplits: 1, omittingEmptySubsequences: false).first.map(String.init) ?? name
        return fingerDigits.contains(first)
    }
}

// MARK: - Gains, bounds, reach

/// The servo constants of the released run.
public struct DecoderGains: Equatable, Sendable {
    /// Control per rad of wanted hinge change: decoder_kp 6.0 (args.json:80 -> train_mimo.py:1414
    /// -> env.py:1294 -> action_decoder.py:258).
    public var kp: Double
    /// Control per rad/s of hinge speed: 0.3 (env.py:991 default; no kd is passed).
    public var kd: Double
    /// Eye servo: gaze_kp 80.0 (args.json:29; env.py:996, :1297).
    public var gazeKp: Double
    /// Eye servo: gaze_kd 3.0 (args.json:30).
    public var gazeKd: Double
    /// Damping of the least squares: 1e-3 (action_decoder.py:259, :190).
    public var damping: Double
    /// POSITION_WEIGHT: 8.0 rad per metre of wanted move (action_decoder.py:139).
    public var positionWeight: Double

    /// Makes a set of gains.
    public init(kp: Double, kd: Double, gazeKp: Double, gazeKd: Double, damping: Double, positionWeight: Double) {
        self.kp = kp
        self.kd = kd
        self.gazeKp = gazeKp
        self.gazeKd = gazeKd
        self.damping = damping
        self.positionWeight = positionWeight
    }

    /// The released run's values: kp 6.0, kd 0.3, gaze kp 80.0, gaze kd 3.0, damping 1e-3,
    /// position weight 8.0.
    public static let released = DecoderGains(kp: 6.0, kd: 0.3, gazeKp: 80.0, gazeKd: 3.0, damping: 1e-3, positionWeight: 8.0)
}

/// How the control bounds are narrowed beyond actuator_ctrlrange (action_decoder.py:280-290:
/// "THE RAILS ARE THE BODY'S, NOT THE MODEL FILE'S": env.py:1296 passes the env action space,
/// which the SpringDamperModel takes from actuator_forcerange, actuation.py:136-138).
public enum ControlBoundsPolicy: Equatable, Sendable {
    /// Narrow to the actuator force range cast to float32, as MIMo's action space holds it
    /// (actuation.py:136; the released behaviour).
    case forceRange
    /// Keep actuator_ctrlrange alone (-1..1 where unlimited).
    case controlRangeOnly
    /// Narrow to the given low and high arrays of nu numbers.
    case explicit(low: [Double], high: [Double])
}

/// Measures how far a handle's own joints can move its body, in metres.
public protocol ReachMeasuring {
    /// The reach of `body` under the chain's `jointIDs`.
    func reachMetres(body: Int, jointIDs: [Int]) throws -> Double
}

/// Port of `_measure_reach` (action_decoder.py:200-241): every joint of the chain set to each
/// end of its range (an unlimited joint to -1.5 and +1.5 rad), the handle body placed by forward
/// kinematics, over all 2^n corners; the reach is the largest distance between any two of those
/// placements, then min(0.5, max(reach, 0.01)); a chain with 0 or more than `maxJoints` joints
/// keeps 0.5 m. Measured on the released body (model_receipt.json motor_synergy.pos_reach_cm):
/// wrists 42.85 cm, fingers 5.48 to 6.80 cm, head 4.81 cm, lower_trunk 5.58 cm.
public struct CornerSweepReach: ReachMeasuring {
    /// The forward-kinematics query.
    public let kinematics: ForwardKinematicsQuerying
    /// The joint tables (ranges and limits).
    public let tables: ActuatedBodyTables
    /// Chains longer than this keep 0.5 m (action_decoder.py:200 max_joints 10).
    public let maxJoints: Int

    /// Makes the measurer.
    public init(kinematics: ForwardKinematicsQuerying, tables: ActuatedBodyTables, maxJoints: Int = 10) {
        self.kinematics = kinematics
        self.tables = tables
        self.maxJoints = maxJoints
    }

    /// See the type's description.
    public func reachMetres(body: Int, jointIDs: [Int]) throws -> Double {
        if jointIDs.isEmpty || jointIDs.count > maxJoints {
            return 0.5
        }
        var points: [Vector3] = []
        let corners = 1 << jointIDs.count
        for corner in 0..<corners {
            var angles: [Int: Double] = [:]
            for (bit, j) in jointIDs.enumerated() {
                let end = (corner >> bit) & 1
                let lo: Double
                let hi: Double
                if j >= 0 && j < tables.jointCount && tables.jointLimited[j] {
                    lo = tables.jointRangeLow[j]
                    hi = tables.jointRangeHigh[j]
                } else {
                    lo = -1.5
                    hi = 1.5
                }
                angles[j] = end == 1 ? hi : lo
            }
            points.append(try kinematics.bodyPosition(ofBody: body, withJointAngles: angles))
        }
        var spread = 0.0
        for a in 0..<points.count {
            for b in 0..<points.count {
                spread = max(spread, (points[a] - points[b]).norm)
            }
        }
        return min(0.5, max(spread, 0.01))
    }
}

/// A fixed reach for every handle: tests, or the receipt's measured values when no kinematics
/// query is injected.
public struct FixedReach: ReachMeasuring, Sendable {
    /// The reach in metres returned for every chain.
    public let metres: Double

    /// Makes the fixed reach.
    public init(metres: Double) {
        self.metres = metres
    }

    /// Returns the fixed value.
    public func reachMetres(body: Int, jointIDs: [Int]) throws -> Double {
        metres
    }
}

// MARK: - The decoder protocol and the handle decoder

/// The array-level decode of action_decoder.py: a command of handle-count x 9 numbers into one
/// control per motor, for one body state. HandleCommandDecoder implements it; CommandDecoder
/// (UEMACore's CommandDecoding) depends on this protocol, so a test can inject a fake.
public protocol ControlDecoding: AnyObject {
    /// nu, the number of controls returned.
    var controlCount: Int { get }
    /// The handle rows a command holds (26 on trunk_v1).
    var handleCount: Int { get }
    /// The contract string the world reports (command_decoder.md open point C).
    var motorContract: String { get }
    /// Decodes one command for the body state of this env step.
    func decode(command: [Double],
                gaze: GazeAngles?,
                handleMask: [Bool]?,
                fingerGoals: [Int: Double]?,
                state: BodyStateReading) throws -> [Double]
}

/// The released decoder: `ActionDecoder` (action_decoder.py:257-520) over the handle layout
/// trunk_v1. Constants built once in `init` (command_decoder.md section 2); `decode` runs the
/// seven steps of command_decoder.md section 3.
public final class HandleCommandDecoder: ControlDecoding {
    /// MOTOR_CONTRACT (action_decoder.py:56).
    public static let motorContractName = "ancestor_arm_axes_signed_limits_v3"
    /// FINGER_SERVO_CONTRACT (action_decoder.py:71).
    public static let fingerServoContractName = "finger_joint_servo_v1"

    /// One handle's constants (`_HandleInfo`, action_decoder.py:244-254).
    public struct HandleInfo: Equatable, Sendable {
        /// The handle's own bone (body id).
        public let bodyID: Int
        /// Every actuated hinge on the chain's bodies, chain sequence then joint-id sequence.
        public let jointIDs: [Int]
        /// jnt_dofadr of each joint.
        public let dofAddresses: [Int]
        /// The first joint-transmission actuator of each joint.
        public let actuatorIDs: [Int]
        /// The chain body each joint sits on.
        public let axisBodyIDs: [Int]
        /// How far the handle's own joints can move it, metres.
        public let reachMetres: Double
    }

    /// One arm solved as a unit (action_decoder.py:312-345).
    public struct ArmGroup: Equatable, Sendable {
        /// Handle indices [wrist, lowerarm, upperarm].
        public let members: [Int]
        /// Per wrist-chain joint, the member whose strength gates that joint's control.
        public let owners: [Int]
        /// Per member, the wrist-chain joint columns whose body is an ancestor (inclusive) of the member's body.
        public let columns: [Int: [Int]]
    }

    /// One eye servo entry (action_decoder.py:347-354): actuator, dof address, gaze component, sign.
    public struct EyeServo: Equatable, Sendable {
        /// The eye joint's actuator.
        public let actuatorID: Int
        /// jnt_dofadr of the eye joint.
        public let dofAddress: Int
        /// 0 azimuth, 1 elevation.
        public let gazeComponent: Int
        /// +1.0 or -1.0.
        public let sign: Double
    }

    /// The tables the decoder was built from.
    public let tables: ActuatedBodyTables
    /// The handle layout.
    public let layout: HandleLayout
    /// The gains.
    public let gains: DecoderGains
    /// Per handle, in row sequence.
    public let handles: [HandleInfo]
    /// The arm groups (two on trunk_v1, none on a layout without the three arm handles of a side).
    public let armGroups: [ArmGroup]
    /// The eye servo entries present and actuated (four on the released body).
    public let eyes: [EyeServo]
    /// Control lower bound per motor.
    public let controlLow: [Double]
    /// Control upper bound per motor.
    public let controlHigh: [Double]
    /// Joint id -> first actuator with a joint transmission on it (action_decoder.py:271-275).
    public let jointToActuator: [Int: Int]
    /// Handle indices whose position row enters the solve.
    public let usePosition: [Bool]
    /// Handle indices of the finger handles (10 on trunk_v1).
    public let fingerHandles: Set<Int>
    /// The controls of the last decode (zeros before the first).
    public private(set) var lastControls: [Double]

    private let grouped: Set<Int>

    /// nu.
    public var controlCount: Int { tables.actuatorCount }

    /// The handle rows a command holds: layout.names.count (26 on trunk_v1).
    public var handleCount: Int { layout.names.count }

    /// MOTOR_CONTRACT + "+" + FINGER_SERVO_CONTRACT (env.py:3576-3583).
    public var motorContract: String {
        HandleCommandDecoder.motorContractName + "+" + HandleCommandDecoder.fingerServoContractName
    }

    /// Builds the constants (action_decoder.py:257-358). Throws when a body or joint the layout
    /// names is missing from the tables, or when an explicit bounds array has the wrong count.
    public init(tables: ActuatedBodyTables,
                layout: HandleLayout = .trunkV1,
                gains: DecoderGains = .released,
                reach: ReachMeasuring,
                bounds: ControlBoundsPolicy = .forceRange) throws {
        self.tables = tables
        self.layout = layout
        self.gains = gains
        let nu = tables.actuatorCount

        var jointToActuator: [Int: Int] = [:]
        for a in 0..<nu where tables.actuatorTransmissionType[a] == ActuatedBodyTables.jointTransmission {
            let j = tables.actuatorTransmissionID[a]
            if jointToActuator[j] == nil {
                jointToActuator[j] = a
            }
        }
        self.jointToActuator = jointToActuator

        var low = [Double](repeating: -1.0, count: nu)
        var high = [Double](repeating: 1.0, count: nu)
        for a in 0..<nu where tables.actuatorControlLimited[a] {
            low[a] = tables.actuatorControlRangeLow[a]
            high[a] = tables.actuatorControlRangeHigh[a]
        }
        switch bounds {
        case .forceRange:
            // MIMo's action space is the force range cast to float32
            // (actuation.py:136 `actuator_forcerange.copy().astype(np.float32)`), which the
            // decoder reads back as float64 (action_decoder.py:287-288): -0.65 becomes
            // -0.6499999761581421. The same rounding is applied here.
            for a in 0..<nu {
                low[a] = max(low[a], Double(Float(tables.actuatorForceRangeLow[a])))
                high[a] = min(high[a], Double(Float(tables.actuatorForceRangeHigh[a])))
            }
        case .controlRangeOnly:
            break
        case let .explicit(explicitLow, explicitHigh):
            if explicitLow.count != nu || explicitHigh.count != nu {
                throw CommandDecoderError.inconsistentTables("explicit bounds have \(explicitLow.count) and \(explicitHigh.count) entries, actuatorCount \(nu)")
            }
            for a in 0..<nu {
                low[a] = max(low[a], explicitLow[a])
                high[a] = min(high[a], explicitHigh[a])
            }
        }
        self.controlLow = low
        self.controlHigh = high

        var handles: [HandleInfo] = []
        for name in layout.names {
            guard let boneName = layout.bones[name], let bodyID = tables.bodyID(named: boneName) else {
                throw CommandDecoderError.missingBody("handle \(name): bone \(layout.bones[name] ?? "(none)")")
            }
            var jointIDs: [Int] = []
            var dofs: [Int] = []
            var acts: [Int] = []
            var axisBodies: [Int] = []
            for chainBody in layout.chains[name] ?? [] {
                guard let b = tables.bodyID(named: chainBody) else {
                    throw CommandDecoderError.missingBody("handle \(name): chain body \(chainBody)")
                }
                for j in 0..<tables.jointCount {
                    if tables.jointBodyID[j] != b || tables.jointType[j] != ActuatedBodyTables.hingeJointType {
                        continue
                    }
                    guard let a = jointToActuator[j] else {
                        continue
                    }
                    jointIDs.append(j)
                    dofs.append(tables.jointDofAddress[j])
                    acts.append(a)
                    axisBodies.append(b)
                }
            }
            let reachMetres = try reach.reachMetres(body: bodyID, jointIDs: jointIDs)
            handles.append(HandleInfo(bodyID: bodyID, jointIDs: jointIDs, dofAddresses: dofs,
                                      actuatorIDs: acts, axisBodyIDs: axisBodies, reachMetres: reachMetres))
        }
        self.handles = handles
        self.usePosition = layout.names.map { layout.positionHandles.contains($0) }
        self.fingerHandles = Set(layout.names.enumerated().filter { layout.isFingerHandle($0.element) }.map { $0.offset })

        var groups: [ArmGroup] = []
        for side in ["L", "R"] {
            guard let wristIndex = layout.index(of: "wrist_\(side)"),
                  let lowerIndex = layout.index(of: "lowerarm_\(side)"),
                  let upperIndex = layout.index(of: "upperarm_\(side)") else {
                continue
            }
            let members = [wristIndex, lowerIndex, upperIndex]
            let wrist = handles[wristIndex]
            var owners: [Int] = []
            for j in wrist.jointIDs {
                let jb = tables.jointBodyID[j]
                var own = members[0]
                for mi in members where handles[mi].bodyID == jb || handles[mi].axisBodyIDs.contains(jb) {
                    own = mi
                }
                for mi in members.dropFirst() where jb == handles[mi].bodyID {
                    own = mi
                }
                owners.append(own)
            }
            var columns: [Int: [Int]] = [:]
            for mi in members {
                var ancestors = Set<Int>()
                var body = handles[mi].bodyID
                var guardCount = 0
                while body > 0 && guardCount <= tables.bodyCount {
                    ancestors.insert(body)
                    body = tables.bodyParentID[body]
                    guardCount += 1
                }
                columns[mi] = wrist.jointIDs.enumerated()
                    .filter { ancestors.contains(tables.jointBodyID[$0.element]) }
                    .map { $0.offset }
            }
            groups.append(ArmGroup(members: members, owners: owners, columns: columns))
        }
        self.armGroups = groups
        self.grouped = Set(groups.flatMap { $0.members })

        var eyes: [EyeServo] = []
        for entry in layout.eyeJoints {
            guard let j = tables.jointID(named: entry.jointName), let a = jointToActuator[j] else {
                continue
            }
            eyes.append(EyeServo(actuatorID: a, dofAddress: tables.jointDofAddress[j],
                                 gazeComponent: entry.gazeComponent, sign: entry.sign))
        }
        self.eyes = eyes
        self.lastControls = [Double](repeating: 0, count: nu)
    }

    /// For reports and tests: the joint names each handle drives (`handle_joint_names`,
    /// action_decoder.py:522-528).
    public func jointNames(forHandle name: String) -> [String] {
        guard let i = layout.index(of: name) else {
            return []
        }
        return handles[i].jointIDs.map { tables.jointNames[$0] }
    }

    // MARK: decode

    /// `decode` (action_decoder.py:402-520). `command`: layout.names.count x 9 numbers,
    /// row-major. `gaze`: the pair as passed to the decoder, or nil (no eye control written).
    /// `handleMask`: nil means every handle active. `fingerGoals`: nil means every chain steps
    /// toward its handle's ordered pose; a dictionary means the finger handles' chains step toward
    /// those hinge goals (finger_joint_servo on). Returns nu controls within controlLow..controlHigh.
    public func decode(command: [Double],
                       gaze: GazeAngles?,
                       handleMask: [Bool]?,
                       fingerGoals: [Int: Double]?,
                       state: BodyStateReading) throws -> [Double] {
        let handleCount = layout.names.count
        if command.count != handleCount * 9 {
            throw CommandDecoderError.commandCount(expected: handleCount * 9, found: command.count)
        }
        if let mask = handleMask, mask.count != handleCount {
            throw CommandDecoderError.maskCount(expected: handleCount, found: mask.count)
        }
        let q = state.qpos
        let qd = state.qvel
        try checkState(q: q, qd: qd)
        let nu = tables.actuatorCount
        var ctrl = [Double](repeating: 0, count: nu)

        // Step 2: every handle outside the arm groups (action_decoder.py:413-436).
        for (i, h) in handles.enumerated() {
            if grouped.contains(i) {
                continue
            }
            if let mask = handleMask, !mask[i] {
                continue
            }
            if h.jointIDs.isEmpty {
                continue
            }
            let strength = clip01(command[i * 9 + 8])
            if strength <= 0.0 {
                continue
            }
            var dq: [Double]
            if let goals = fingerGoals, fingerHandles.contains(i) {
                dq = try fingerJointStep(handle: h, goals: goals, q: q)
            } else {
                dq = try poseStep(command: command, handleIndex: i, handle: h, state: state)
            }
            for k in 0..<h.jointIDs.count {
                let j = h.jointIDs[k]
                if tables.jointLimited[j] {
                    let qj = q[tables.jointQposAddress[j]]
                    dq[k] = min(max(qj + dq[k], tables.jointRangeLow[j]), tables.jointRangeHigh[j]) - qj
                }
                let u = strength * (gains.kp * dq[k] - gains.kd * qd[h.dofAddresses[k]])
                ctrl[h.actuatorIDs[k]] += u
            }
        }

        // Step 3: each arm as one stacked system (action_decoder.py:437-482).
        for group in armGroups {
            let wrist = handles[group.members[0]]
            if wrist.jointIDs.isEmpty {
                continue
            }
            let active = group.members.filter { mi in
                (handleMask.map { $0[mi] } ?? true) && command[mi * 9 + 8] > 0.0
            }
            if active.isEmpty {
                continue
            }
            let joints = wrist.jointIDs
            let nJ = joints.count
            let axes = joints.map { state.jointAxis($0) }
            var rows: [[Double]] = []
            var rhs: [Double] = []
            for mi in active {
                let h = handles[mi]
                let sm = clip01(command[mi * 9 + 8])
                let targetRotation = Matrix3(quaternionXYZW: Array(command[(mi * 9 + 3)..<(mi * 9 + 7)]))
                let currentRotation = state.bodyRotation(h.bodyID)
                let w = (targetRotation * currentRotation.transposed).logarithm()
                let cols = group.columns[mi] ?? []
                var a = [[Double]](repeating: [Double](repeating: 0, count: nJ), count: 3)
                for k in cols {
                    a[0][k] = sm * axes[k].x
                    a[1][k] = sm * axes[k].y
                    a[2][k] = sm * axes[k].z
                }
                rows.append(contentsOf: a)
                rhs.append(contentsOf: [sm * w.x, sm * w.y, sm * w.z])
                let target = Vector3(command[mi * 9], command[mi * 9 + 1], command[mi * 9 + 2])
                if usePosition[mi] && target.isFinite {
                    let pBody = state.bodyPosition(h.bodyID)
                    let e = ((1.0 / 100.0) * target - pBody).clipped(to: h.reachMetres)
                    var jac = [[Double]](repeating: [Double](repeating: 0, count: nJ), count: 3)
                    for k in cols {
                        let lever = axes[k].cross(pBody - state.jointAnchor(joints[k]))
                        jac[0][k] = sm * gains.positionWeight * lever.x
                        jac[1][k] = sm * gains.positionWeight * lever.y
                        jac[2][k] = sm * gains.positionWeight * lever.z
                    }
                    rows.append(contentsOf: jac)
                    let s = sm * gains.positionWeight
                    rhs.append(contentsOf: [s * e.x, s * e.y, s * e.z])
                }
            }
            var dq = try DampedLeastSquares.solve(rows: rows, rhs: rhs, columns: nJ, damping: gains.damping)
            let largestActiveStrength = active.map { clip01(command[$0 * 9 + 8]) }.max() ?? 0.0
            for k in 0..<nJ {
                let j = joints[k]
                if tables.jointLimited[j] {
                    let qj = q[tables.jointQposAddress[j]]
                    dq[k] = min(max(qj + dq[k], tables.jointRangeLow[j]), tables.jointRangeHigh[j]) - qj
                }
                let owner = group.owners[k]
                let sOwn = active.contains(owner) ? clip01(command[owner * 9 + 8]) : largestActiveStrength
                ctrl[wrist.actuatorIDs[k]] += sOwn * (gains.kp * dq[k] - gains.kd * qd[wrist.dofAddresses[k]])
            }
        }

        // Step 4: direction-preserving rescale per chain and per arm group (action_decoder.py:483-506).
        var groupsToScale: [[Int]] = handles.filter { !$0.actuatorIDs.isEmpty }.map { $0.actuatorIDs }
        for group in armGroups {
            groupsToScale.append(handles[group.members[0]].actuatorIDs)
        }
        for ids in groupsToScale where !ids.isEmpty {
            var smallestRatio = Double.infinity
            for a in ids {
                let value = ctrl[a]
                if value == 0 {
                    smallestRatio = min(smallestRatio, 1.0)
                    continue
                }
                let bound = value >= 0 ? controlHigh[a] : -controlLow[a]
                smallestRatio = min(smallestRatio, bound / abs(value))
            }
            let factor = max(0.0, min(1.0, smallestRatio))
            for a in ids { ctrl[a] *= factor }
        }

        // Step 5: the clip (action_decoder.py:507).
        for a in 0..<nu {
            ctrl[a] = min(max(ctrl[a], controlLow[a]), controlHigh[a])
        }

        // Step 6: the eye servo, SET not added (action_decoder.py:508-518).
        if let gaze = gaze {
            let pair = [gaze.azimuth, gaze.elevation]
            for eye in eyes {
                let j = tables.actuatorTransmissionID[eye.actuatorID]
                let adr = tables.jointQposAddress[j]
                var target = eye.sign * pair[eye.gazeComponent]
                if tables.jointLimited[j] {
                    target = min(max(target, tables.jointRangeLow[j]), tables.jointRangeHigh[j])
                }
                let u = gains.gazeKp * (target - q[adr]) - gains.gazeKd * qd[eye.dofAddress]
                ctrl[eye.actuatorID] = min(max(u, controlLow[eye.actuatorID]), controlHigh[eye.actuatorID])
            }
        }

        lastControls = ctrl
        return ctrl
    }

    // MARK: one chain's step

    /// `_pose_step` (action_decoder.py:361-389): the chain's dq (rad per hinge) toward the
    /// handle's ordered pose.
    private func poseStep(command: [Double], handleIndex i: Int, handle h: HandleInfo, state: BodyStateReading) throws -> [Double] {
        let targetRotation = Matrix3(quaternionXYZW: Array(command[(i * 9 + 3)..<(i * 9 + 7)]))
        let currentRotation = state.bodyRotation(h.bodyID)
        let w = (targetRotation * currentRotation.transposed).logarithm()
        let nJ = h.jointIDs.count
        let axes = h.jointIDs.map { state.jointAxis($0) }
        var rows = [[Double]](repeating: [Double](repeating: 0, count: nJ), count: 3)
        for k in 0..<nJ {
            rows[0][k] = axes[k].x
            rows[1][k] = axes[k].y
            rows[2][k] = axes[k].z
        }
        var rhs = [w.x, w.y, w.z]
        let target = Vector3(command[i * 9], command[i * 9 + 1], command[i * 9 + 2])
        if usePosition[i] && target.isFinite {
            let pBody = state.bodyPosition(h.bodyID)
            let e = ((1.0 / 100.0) * target - pBody).clipped(to: h.reachMetres)
            var jac = [[Double]](repeating: [Double](repeating: 0, count: nJ), count: 3)
            for k in 0..<nJ {
                let lever = axes[k].cross(pBody - state.jointAnchor(h.jointIDs[k]))
                jac[0][k] = gains.positionWeight * lever.x
                jac[1][k] = gains.positionWeight * lever.y
                jac[2][k] = gains.positionWeight * lever.z
            }
            rows.append(contentsOf: jac)
            rhs.append(contentsOf: [gains.positionWeight * e.x, gains.positionWeight * e.y, gains.positionWeight * e.z])
        }
        return try DampedLeastSquares.solve(rows: rows, rhs: rhs, columns: nJ, damping: gains.damping)
    }

    /// `_finger_joint_step` (action_decoder.py:391-399): each hinge's goal minus its angle; a
    /// hinge without a goal throws, naming the joint ids.
    private func fingerJointStep(handle h: HandleInfo, goals: [Int: Double], q: [Double]) throws -> [Double] {
        let missing = h.jointIDs.filter { goals[$0] == nil }
        if !missing.isEmpty {
            throw CommandDecoderError.missingFingerGoal(missing)
        }
        return h.jointIDs.map { j in (goals[j] ?? 0.0) - q[tables.jointQposAddress[j]] }
    }

    /// Every qpos and qvel address the tables use must exist in the state.
    private func checkState(q: [Double], qd: [Double]) throws {
        for h in handles {
            for (k, j) in h.jointIDs.enumerated() {
                if tables.jointQposAddress[j] >= q.count {
                    throw CommandDecoderError.stateTooShort("qpos has \(q.count) entries, joint \(j) needs address \(tables.jointQposAddress[j])")
                }
                if h.dofAddresses[k] >= qd.count {
                    throw CommandDecoderError.stateTooShort("qvel has \(qd.count) entries, joint \(j) needs address \(h.dofAddresses[k])")
                }
            }
        }
        for eye in eyes {
            let j = tables.actuatorTransmissionID[eye.actuatorID]
            if tables.jointQposAddress[j] >= q.count || eye.dofAddress >= qd.count {
                throw CommandDecoderError.stateTooShort("eye joint \(j) address outside qpos \(q.count) or qvel \(qd.count)")
            }
        }
    }

    private func clip01(_ v: Double) -> Double {
        min(max(v, 0.0), 1.0)
    }
}

// MARK: - The body state over UEMACore's PhysicsReading

/// UEMACore's PhysicsReading flat arrays (xpos nbody x 3, xmat nbody x 9, xaxis and xanchor
/// njnt x 3, qpos nq, qvel nv; Protocols.swift's PhysicsReading) read as the decoder's
/// per-body and per-joint values. The six arrays are copied once at init, so one decode reads
/// one consistent env step. A body or joint index outside its array gives the zero vector (or
/// the identity rotation); HandleCommandDecoder's own check covers the qpos and qvel addresses.
public struct PhysicsBodyState: BodyStateReading, Sendable {
    /// data.qpos, nq numbers.
    public let qpos: [Double]
    /// data.qvel, nv numbers.
    public let qvel: [Double]
    private let xpos: [Double]
    private let xmat: [Double]
    private let xaxis: [Double]
    private let xanchor: [Double]

    /// Copies the arrays of the physics at this env step.
    public init(physics: any PhysicsReading) {
        self.init(qpos: physics.qpos, qvel: physics.qvel, xpos: physics.xpos,
                  xmat: physics.xmat, xaxis: physics.xaxis, xanchor: physics.xanchor)
    }

    /// Copies the arrays of the physics at this env step and checks each count against the
    /// physics's own sizes (qpos nq, qvel nv, xpos nbody x 3, xmat nbody x 9, xaxis and xanchor
    /// njnt x 3), so that a short array throws CommandDecoderError.stateTooShort naming it
    /// instead of reading as the zero vector or the identity rotation.
    public init(validating physics: any PhysicsReading) throws {
        let sizes = physics.sizes
        self.init(physics: physics)
        let checks: [(String, Int, Int)] = [
            ("qpos", qpos.count, sizes.nq), ("qvel", qvel.count, sizes.nv),
            ("xpos", xpos.count, sizes.nbody * 3), ("xmat", xmat.count, sizes.nbody * 9),
            ("xaxis", xaxis.count, sizes.njnt * 3), ("xanchor", xanchor.count, sizes.njnt * 3),
        ]
        for (name, found, expected) in checks where found != expected {
            throw CommandDecoderError.stateTooShort("\(name) has \(found) numbers, the physics sizes give \(expected)")
        }
    }

    /// Wraps the six flat arrays as given (tests, fixtures).
    public init(qpos: [Double], qvel: [Double], xpos: [Double], xmat: [Double], xaxis: [Double], xanchor: [Double]) {
        self.qpos = qpos
        self.qvel = qvel
        self.xpos = xpos
        self.xmat = xmat
        self.xaxis = xaxis
        self.xanchor = xanchor
    }

    /// xpos[body], metres.
    public func bodyPosition(_ body: Int) -> Vector3 {
        Vector3(array: PhysicsBodyState.row(xpos, index: body, width: 3))
    }

    /// xmat[body], row-major.
    public func bodyRotation(_ body: Int) -> Matrix3 {
        Matrix3(rowMajor: PhysicsBodyState.row(xmat, index: body, width: 9))
    }

    /// xaxis[joint].
    public func jointAxis(_ joint: Int) -> Vector3 {
        Vector3(array: PhysicsBodyState.row(xaxis, index: joint, width: 3))
    }

    /// xanchor[joint], metres.
    public func jointAnchor(_ joint: Int) -> Vector3 {
        Vector3(array: PhysicsBodyState.row(xanchor, index: joint, width: 3))
    }

    /// Row `index` of a flat array of rows of `width` numbers; an empty array when outside.
    private static func row(_ values: [Double], index: Int, width: Int) -> [Double] {
        let start = index * width
        guard index >= 0, start + width <= values.count else {
            return []
        }
        return Array(values[start..<(start + width)])
    }
}

// MARK: - UEMACore's CommandDecoding

/// The command decoder of UEMACore's contract (CommandDecoding; specification REQ-B-13): one
/// command at `.centimetres` scale, the handle mask, the gaze pair and the 44 finger joint
/// goals into the 90 motor controls of the current env step. It converts UEMACore's value
/// types into the arrays of the injected ControlDecoding and reads the body state from the
/// injected PhysicsReading; the arithmetic is the engine's.
public final class CommandDecoder: CommandDecoding {
    /// The 44 hand hinge joints in the geometry sequence FingerJointGoals uses: the receipt's
    /// `hand_response.joint_names` (weights/b1_continue_B1K_skip13_492800/run/checkpoints/
    /// ckpt_step_494025/model_receipt.json, 44 names, read November 4, 2026); the Python maps
    /// goal k to the k-th joint's MuJoCo id (`goals_by_joint`, hand_target_control.py:457-463).
    public static let fingerJointNames: [String] = [
        "robot:left_ff_side", "robot:left_ff_knuckle", "robot:left_ff_middle", "robot:left_ff_distal",
        "robot:left_mf_side", "robot:left_mf_knuckle", "robot:left_mf_middle", "robot:left_mf_distal",
        "robot:left_rf_side", "robot:left_rf_knuckle", "robot:left_rf_middle", "robot:left_rf_distal",
        "robot:left_lf_meta", "robot:left_lf_side", "robot:left_lf_knuckle", "robot:left_lf_middle",
        "robot:left_lf_distal",
        "robot:left_th_swivel", "robot:left_th_adduction", "robot:left_th_pivot", "robot:left_th_middle",
        "robot:left_th_distal",
        "robot:right_ff_side", "robot:right_ff_knuckle", "robot:right_ff_middle", "robot:right_ff_distal",
        "robot:right_mf_side", "robot:right_mf_knuckle", "robot:right_mf_middle", "robot:right_mf_distal",
        "robot:right_rf_side", "robot:right_rf_knuckle", "robot:right_rf_middle", "robot:right_rf_distal",
        "robot:right_lf_meta", "robot:right_lf_side", "robot:right_lf_knuckle", "robot:right_lf_middle",
        "robot:right_lf_distal",
        "robot:right_th_swivel", "robot:right_th_adduction", "robot:right_th_pivot", "robot:right_th_middle",
        "robot:right_th_distal",
    ]

    /// The MuJoCo joint ids of `fingerJointNames` in `tables`; a name not in the tables throws
    /// CommandDecoderError.missingBody naming it.
    public static func fingerJointIDs(in tables: ActuatedBodyTables) throws -> [Int] {
        try fingerJointNames.map { (name: String) throws -> Int in
            guard let j = tables.jointID(named: name) else {
                throw CommandDecoderError.missingBody("finger joint \(name)")
            }
            return j
        }
    }

    /// The array-level decoder.
    public let engine: ControlDecoding
    /// The joint id of each of the 44 finger joint goals, in FingerJointGoals sequence.
    public let fingerJointIDs: [Int]
    /// The controls of the last decode (all 0 before the first).
    public private(set) var lastControls: MotorControls

    private let physics: any PhysicsReading

    /// Makes the decoder. Throws CommandDecoderError.inconsistentTables when the engine's
    /// handle rows are not 26, its controls not MotorControls.count (90), or `fingerJointIDs` not 44 entries.
    public init(engine: ControlDecoding, physics: any PhysicsReading, fingerJointIDs: [Int]) throws {
        if engine.handleCount != Command.handleCount {
            throw CommandDecoderError.inconsistentTables("engine has \(engine.handleCount) handle rows, Command \(Command.handleCount)")
        }
        if engine.controlCount != MotorControls.count {
            throw CommandDecoderError.inconsistentTables("engine has \(engine.controlCount) controls, MotorControls \(MotorControls.count)")
        }
        if fingerJointIDs.count != FingerJointGoals.count {
            throw CommandDecoderError.inconsistentTables("fingerJointIDs has \(fingerJointIDs.count) entries, FingerJointGoals \(FingerJointGoals.count)")
        }
        self.engine = engine
        self.physics = physics
        self.fingerJointIDs = fingerJointIDs
        self.lastControls = MotorControls.zero
    }

    /// MOTOR_CONTRACT + "+" + FINGER_SERVO_CONTRACT, from the engine.
    public var motorContract: String { engine.motorContract }

    /// See UEMACore's CommandDecoding. `gaze` nil takes `command.gazeAngles` (row 23);
    /// `fingerGoals` nil decodes the finger rows from their quaternions and positions.
    public func decode(_ command: Command, mask: HandleMask, gaze: GazeAngles?, fingerGoals: FingerJointGoals?) throws -> MotorControls {
        guard command.scale == .centimetres else {
            throw UEMACoreError.wrongScale(expected: .centimetres, found: command.scale)
        }
        let values = command.values.map { Double($0) }
        var goals: [Int: Double]?
        if let fingerGoals = fingerGoals {
            var byJoint: [Int: Double] = [:]
            for (k, j) in fingerJointIDs.enumerated() {
                byJoint[j] = fingerGoals.angles[k]
            }
            goals = byJoint
        }
        let ctrl = try engine.decode(command: values,
                                     gaze: gaze ?? command.gazeAngles,
                                     handleMask: mask.values,
                                     fingerGoals: goals,
                                     state: PhysicsBodyState(validating: physics))
        let controls = try MotorControls(values: ctrl)
        lastControls = controls
        return controls
    }
}

// MARK: - One env step: decode, modulate, write through Physics

/// One env step of the Python world's `step` (env.py:2884-2893), in its order: the decode of
/// the command for the body state of this env step, then the body-function modulation
/// (MotorFunction.swift) when one is injected, then the write of the controls through UEMACore's
/// Physics and its 2 physics steps of 5 ms (`Physics.stepEnvStep`). The session calls it 5
/// times per model step with the same command (command_decoder.md section 1).
public final class MotorCommandWriter {
    private let decoder: CommandDecoding
    private let physics: any Physics
    private let motorFunction: MotorFunctionApplier?

    /// Makes the writer; `motorFunction` nil leaves the actuator tables native (`body_function off`).
    public init(decoder: CommandDecoding, physics: any Physics, motorFunction: MotorFunctionApplier?) {
        self.decoder = decoder
        self.physics = physics
        self.motorFunction = motorFunction
    }

    /// Decodes, modulates and writes the controls into data.ctrl without stepping.
    @discardableResult
    public func write(_ command: Command, mask: HandleMask, gaze: GazeAngles?, fingerGoals: FingerJointGoals?) throws -> MotorControls {
        let controls = try decoder.decode(command, mask: mask, gaze: gaze, fingerGoals: fingerGoals)
        try motorFunction?.apply()
        try physics.write(controls: controls)
        return controls
    }

    /// Decodes, modulates, then runs one env step (write and 2 physics steps) through Physics.
    @discardableResult
    public func stepEnvStep(_ command: Command,
                            mask: HandleMask,
                            gaze: GazeAngles?,
                            fingerGoals: FingerJointGoals?,
                            afterEachPhysicsStep: (any PhysicsReading) throws -> Void) throws -> MotorControls {
        let controls = try decoder.decode(command, mask: mask, gaze: gaze, fingerGoals: fingerGoals)
        try motorFunction?.apply()
        try physics.stepEnvStep(controls: controls, afterEachPhysicsStep: afterEachPhysicsStep)
        return controls
    }
}

// MARK: - Adapters over UEMAPhysics (WI-25)

extension ActuatedBodyTables {
    /// The tables from UEMAPhysics's PhysicsModelTables, which copies the same mjModel fields
    /// under the same names (Sources/UEMAPhysics/PhysicsState.swift). Read the tables before the
    /// first body-function write: PhysicsModelTables holds actuator_gainprm and
    /// actuator_forcerange as they are at the copy, and both the decoder's bounds and the
    /// modulation's native values must be the scene's own (env.py:2837-2839). Throws as the
    /// memberwise initializer does.
    public init(modelTables t: PhysicsModelTables) throws {
        try self.init(actuatorCount: t.actuatorCount,
                      bodyNames: t.bodyNames,
                      bodyParentID: t.bodyParentID,
                      jointNames: t.jointNames,
                      jointType: t.jointType,
                      jointBodyID: t.jointBodyID,
                      jointQposAddress: t.jointQposAddress,
                      jointDofAddress: t.jointDofAddress,
                      jointLimited: t.jointLimited,
                      jointRangeLow: t.jointRangeLow,
                      jointRangeHigh: t.jointRangeHigh,
                      actuatorTransmissionType: t.actuatorTransmissionType,
                      actuatorTransmissionID: t.actuatorTransmissionID,
                      actuatorControlLimited: t.actuatorControlLimited,
                      actuatorControlRangeLow: t.actuatorControlRangeLow,
                      actuatorControlRangeHigh: t.actuatorControlRangeHigh,
                      actuatorForceRangeLow: t.actuatorForceRangeLow,
                      actuatorForceRangeHigh: t.actuatorForceRangeHigh,
                      actuatorGainType: t.actuatorGainType,
                      actuatorBiasType: t.actuatorBiasType,
                      actuatorGainParameter0: t.actuatorGainParameter0)
    }
}

/// ForwardKinematicsQuerying over UEMAPhysics's ScratchKinematicsQuerying: the scratch mjData
/// query of action_decoder.py:224-238 (qpos at qpos0, the listed joints at their angles,
/// mj_forward, xpos[body] read), its 3 numbers returned as a Vector3 in metres.
public struct ScratchForwardKinematics: ForwardKinematicsQuerying {
    private let query: any ScratchKinematicsQuerying

    /// Wraps the scratch query (MuJoCoPhysics on the real scene).
    public init(query: any ScratchKinematicsQuerying) {
        self.query = query
    }

    /// The body's world position; a result of other than 3 numbers throws
    /// CommandDecoderError.inconsistentTables naming the count.
    public func bodyPosition(ofBody body: Int, withJointAngles jointAngles: [Int: Double]) throws -> Vector3 {
        let p = try query.scratchBodyPosition(ofBody: body, withJointAngles: jointAngles)
        guard p.count == 3 else {
            throw CommandDecoderError.inconsistentTables("scratch position of body \(body) has \(p.count) numbers, expected 3")
        }
        return Vector3(p[0], p[1], p[2])
    }
}

extension CommandDecoder {
    /// The released decoder over one loaded physics: ActuatedBodyTables from its model tables,
    /// HandleCommandDecoder with layout trunk_v1, the released gains (kp 6.0, kd 0.3, gaze kp
    /// 80.0, gaze kd 3.0, damping 1e-3, position weight 8.0), bounds narrowed to the float32
    /// force range, the reach measured by CornerSweepReach over the scratch kinematics, and the
    /// 44 finger joint ids looked up by name. Throws CommandDecoderError.missingBody for a body
    /// or finger joint the scene lacks, and CommandDecoderError.inconsistentTables when the
    /// scene's nu is not MotorControls.count (THE MOTOR COUNT, this file's header).
    public static func released<P: PhysicsReading & PhysicsModelDescribing & ScratchKinematicsQuerying>(
        physics: P
    ) throws -> CommandDecoder {
        let tables = try ActuatedBodyTables(modelTables: physics.modelTables())
        let reach = CornerSweepReach(kinematics: ScratchForwardKinematics(query: physics), tables: tables)
        let engine = try HandleCommandDecoder(tables: tables, reach: reach)
        return try CommandDecoder(engine: engine, physics: physics, fingerJointIDs: CommandDecoder.fingerJointIDs(in: tables))
    }
}
