// Command.swift: the command (26 body handles x 9 numbers), its handle and column names,
// the gaze angles of the eyes' row, the handle mask, the 44 finger joint goals and the
// 90 motor controls the decoder makes of a command. Work item WI-20, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). Foundation is imported for atan2, asin, sin and cos.
// The "First build on a Mac" checklist of this work item is in Protocols.swift's header.
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave:
// - README.md lines 19-20: "A command is 9 numbers for each of 26 body handles (target
//   position, rotation, radius, strength); handle 23 is the eyes: gaze is a limb here, not
//   a separate system."
// - apple/UEMAKit/Specs/command_decoder.md section 1, the `motor` input: "[26, 9] float64.
//   Columns 0-2 target position, cm, world frame; 3-6 target orientation as a quaternion
//   x y z w, world frame; 7 radius (never read, action_decoder.py:43 "Radius is ignored");
//   8 strength, clipped to [0, 1] at line 420. Rows follow `HANDLE_NAMES` (24,
//   lem/model/body_sets.py:52-75) then `lower_trunk`, `chest` (mimo_bridge/body_layout.py:6-14,
//   layout `trunk_v1`, args.json:140)."
// - Source/Python/lem/model/body_sets.py lines 52-75: HANDLE_NAMES = ("wrist_L", "thumb_L",
//   "index_L", "middle_L", "ring_L", "pinky_L", "wrist_R", "thumb_R", "index_R", "middle_R",
//   "ring_R", "pinky_R", "head", "upperarm_L", "lowerarm_L", "upperarm_R", "lowerarm_R",
//   "thigh_L", "calf_L", "thigh_R", "calf_R", "foot_L", "foot_R", "gaze"); line 45:
//   "NUM_HANDLES: int = 24".
// - Source/Python/mimo_bridge/body_layout.py lines 6-14: "TRUNK_NAMES = ('lower_trunk',
//   'chest')"; "if layout == 'trunk_v1':" / "return HANDLE_NAMES + TRUNK_NAMES".
// - Source/Python/lem/model/encoder_registry.py lines 242-267: "COMMAND_POSITION_RANGE_CM =
//   200.0"; normalize_executed_command: "out[..., 0:3] = out[..., 0:3] /
//   float(position_range_cm)"; "NOT for the controller's raw output: motor_out positions are
//   already normalized."
// - apple/UEMAKit/Specs/motor_controller_input.md section 4, "Output units": "position in
//   position_range units (x 200 = cm after the execution transform), quaternion xyzw,
//   radius 0-1, strength 0-1".
// - Source/Python/mimo_bridge/lem_model.py lines 5122-5140 (_gaze_from_handle): the
//   quaternion of handle 23, "q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)"; "yaw =
//   torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))"; "pitch = torch.asin((2 *
//   (w * y - z * x)).clamp(-1.0, 1.0))"; "az = yaw.clamp(*AZ_RANGE)"; elevation = pitch
//   under eye_row_convention 'legacy' (args.json:215 per command_decoder.md section 1);
//   "el = el.clamp(*EL_RANGE)". Lines 5094-5107 (gaze_to_handle, the inverse): "ca, sa =
//   np.cos(az / 2.0), np.sin(az / 2.0)"; "ce, se = np.cos(pitch / 2.0), np.sin(pitch / 2.0)";
//   "out[:, gi, 3] = -sa * se      # x"; "out[:, gi, 4] = ca * se       # y";
//   "out[:, gi, 5] = sa * ce       # z"; "out[:, gi, 6] = ca * ce       # w"; line 5073:
//   "GAZE_IDX = HANDLE_NAMES.index("gaze")" (= 23).
// - Source/Python/mimo_bridge/gaze_reflex.py lines 41-42: "AZ_RANGE = (-0.785, 0.785)
//   # +/- 45 deg"; "EL_RANGE = (-0.820, 0.576)     # -47 to +33 deg".
// - apple/UEMAKit/Specs/command_decoder.md section 1: `handle_mask` "[26] bool. Every
//   curriculum phase's mask is all True (built by lem/training/curriculum.py:236-246;
//   lem_model.py:5720-5721 records that all six phases' masks are all True)"; `finger_goals` "the 44
//   angles in the geometry sequence (the receipt's `hand_response.joint_names` ... 44
//   names)"; section 2: `POSITION_HANDLES` "the 12 hand handles (wrists and finger tips)
//   plus `head`, `lower_trunk`: 14 handles"; `finger_handles` "handle indices whose name
//   starts with a `FINGER_DIGITS` word (10 of them)"; `ctrl_low`, `ctrl_high` from
//   `actuator_ctrlrange` "else -1, 1"; section 3 step 5: "`ctrl = clip(ctrl, ctrl_low,
//   ctrl_high)`"; section 4: "`ctrl`: [nu] float64, nu = `model.nu`".
// - docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html row WI-25: "writes ctrl
//   clipped to -1..1". The row's evidence "92 motors ctrlrange -1 1 (inventory2 mimo-body)"
//   counts a different body file: the released scene
//   (Sources/UEMAScene/Resources/crib_age24_trunk_v1/scene.xml) compiles to nu = 90 in Python
//   mujoco 3.3.0 (measured November 5, 2026: 90 actuators, every one a joint transmission on a
//   hinge joint) and in the Linux build of MuJoCoPhysics (MuJoCoPhysicsTests
//   testLoadsReleasedSceneWithMeasuredSizes); bundle.json model_counts.nu is 90.
// - apple/UEMAKit/Specs/proprio_rows.md section 5 step 1: "`echo[:, 9] = handle_mask` as
//   1.0/0.0 (1.0 when no mask)".
// - The round trip of `gazeAngles` after `setGaze` was checked here with numpy on a 25 x 25
//   grid over both angle ranges: largest difference 3.3e-16 rad.

import Foundation

// MARK: - Errors

/// The one error type of this module's value types: a count or shape that does not match
/// the released model's fixed sizes, an index outside a fixed range, a missing sense, or a
/// command of the wrong scale.
public enum UEMACoreError: Error, Equatable, Sendable {
    /// `type` was given `found` numbers where it holds exactly `expected`.
    case wrongCount(type: String, expected: Int, found: Int)
    /// `type` was given the shape `found` where `expected` was required.
    case wrongShape(type: String, expected: [Int], found: [Int])
    /// A dimension of `shape` is negative, or the product of `shape` is not `valueCount`.
    case invalidShape(type: String, shape: [Int], valueCount: Int)
    /// `index` lies outside 0 ..< `count` of `type`.
    case indexOutOfRange(type: String, index: Int, count: Int)
    /// A token set or input set lacks this active sense.
    case missingSense(SenseName)
    /// A command of scale `found` was given where `expected` was required.
    case wrongScale(expected: CommandScale, found: CommandScale)
    /// A vision reading lacks the image of this camera (its scene XML name).
    case missingCamera(String)
}

// MARK: - Handles and columns

/// One of the 26 body handles a command addresses, in the row sequence of the released body
/// layout `trunk_v1`: the 24 of HANDLE_NAMES (body_sets.py:52-75) then lower_trunk and chest
/// (body_layout.py:6-14). The raw value is the row index in the command.
public enum BodyHandle: Int, CaseIterable, Codable, Hashable, Sendable {
    case wristL = 0
    case thumbL = 1
    case indexL = 2
    case middleL = 3
    case ringL = 4
    case pinkyL = 5
    case wristR = 6
    case thumbR = 7
    case indexR = 8
    case middleR = 9
    case ringR = 10
    case pinkyR = 11
    case head = 12
    case upperarmL = 13
    case lowerarmL = 14
    case upperarmR = 15
    case lowerarmR = 16
    case thighL = 17
    case calfL = 18
    case thighR = 19
    case calfR = 20
    case footL = 21
    case footR = 22
    /// Row 23, the eyes: its quaternion is the gaze direction (README.md:19-20); its
    /// position, radius and strength are never read by the body (lem_model.py:5125-5127).
    case gaze = 23
    case lowerTrunk = 24
    case chest = 25

    /// The number of handles: 26.
    public static let count = 26

    /// The Python names, by row: HANDLE_NAMES (body_sets.py:52-75) then TRUNK_NAMES
    /// (body_layout.py:6).
    public static let pythonNames: [String] = [
        "wrist_L", "thumb_L", "index_L", "middle_L", "ring_L", "pinky_L",
        "wrist_R", "thumb_R", "index_R", "middle_R", "ring_R", "pinky_R",
        "head",
        "upperarm_L", "lowerarm_L",
        "upperarm_R", "lowerarm_R",
        "thigh_L", "calf_L",
        "thigh_R", "calf_R",
        "foot_L", "foot_R",
        "gaze",
        "lower_trunk", "chest",
    ]

    /// This handle's Python name, such as "wrist_L" or "lower_trunk".
    public var pythonName: String { BodyHandle.pythonNames[rawValue] }

    /// The handle with this Python name; nil for a name that is not one of the 26.
    public init?(pythonName: String) {
        guard let index = BodyHandle.pythonNames.firstIndex(of: pythonName),
              let handle = BodyHandle(rawValue: index) else {
            return nil
        }
        self = handle
    }

    /// True for the 10 finger handles (thumb to pinky on each side): the decoder's
    /// `finger_handles` (command_decoder.md section 2), whose rows the finger joint goals
    /// replace under `finger_joint_servo on`.
    public var isFinger: Bool {
        (1 ... 5).contains(rawValue) || (7 ... 11).contains(rawValue)
    }

    /// The 10 finger handles in row sequence.
    public static let fingers: [BodyHandle] = allCases.filter { $0.isFinger }

    /// The 14 handles whose position row enters the decoder's solve: the 12 hand handles,
    /// head and lower_trunk (`POSITION_HANDLES`, command_decoder.md section 2).
    public static let positionHandles: [BodyHandle] = [
        .wristL, .thumbL, .indexL, .middleL, .ringL, .pinkyL,
        .wristR, .thumbR, .indexR, .middleR, .ringR, .pinkyR,
        .head, .lowerTrunk,
    ]
}

/// One of the 9 numbers of a command row (command_decoder.md section 1): target position
/// x y z (columns 0-2), target orientation as a quaternion x y z w (3-6), radius (7, never
/// read by the decoder) and strength (8, clipped to 0 to 1). The raw value is the column.
public enum CommandColumn: Int, CaseIterable, Codable, Hashable, Sendable {
    case positionX = 0
    case positionY = 1
    case positionZ = 2
    case quaternionX = 3
    case quaternionY = 4
    case quaternionZ = 5
    case quaternionW = 6
    case radius = 7
    case strength = 8

    /// The number of columns: 9.
    public static let count = 9

    /// The three position columns.
    public static let position: [CommandColumn] = [.positionX, .positionY, .positionZ]

    /// The four quaternion columns, x y z w.
    public static let quaternion: [CommandColumn] = [
        .quaternionX, .quaternionY, .quaternionZ, .quaternionW,
    ]
}

/// The unit of a command's three position columns. The other six columns (quaternion,
/// radius, strength) are dimensionless at both scales.
public enum CommandScale: String, Codable, Hashable, Sendable {
    /// Position in position_range units: cm / 200 (`COMMAND_POSITION_RANGE_CM` 200.0,
    /// encoder_registry.py:242). The motor controller's output, the planner's raw action
    /// and the command columns of the motor encoder's input are at this scale
    /// (motor_controller_input.md section 4; proprio_rows.md section 5).
    case normalized
    /// Position in centimetres, world frame: the executed command the decoder reads
    /// (command_decoder.md section 1), reached from `normalized` through the execution
    /// transform (planner_loss.md section 3 step 2), which is the session's step, not a
    /// multiplication by 200.
    case centimetres
}

// MARK: - The command

/// The command: 26 body handles x 9 numbers = 234 float32 numbers (README.md:19-20),
/// row-major (`values[handle.rawValue * 9 + column.rawValue]`), tagged with the scale of its
/// position columns. The motor controller emits one every model step at `.normalized`
/// scale; the planner refines one every 10th model step; the decoder reads one at
/// `.centimetres` scale and turns it into 90 motor controls (`MotorControls`).
public struct Command: Equatable, Hashable, Codable, Sendable {
    /// The handles per command: 26.
    public static let handleCount = BodyHandle.count

    /// The numbers per handle: 9.
    public static let numbersPerHandle = CommandColumn.count

    /// The numbers per command: 26 x 9 = 234.
    public static let count = 234

    /// Centimetres per position_range unit: 200 (`COMMAND_POSITION_RANGE_CM`,
    /// encoder_registry.py:242).
    public static let positionRangeCentimetres: Float = 200

    /// The eyes' handle, row 23 (lem_model.py:5073).
    public static let gazeHandle = BodyHandle.gaze

    /// The scale of the three position columns of every row.
    public let scale: CommandScale

    /// The 234 numbers, row-major by handle then column.
    public private(set) var values: [Float]

    /// Wraps exactly 234 numbers; any other count throws `UEMACoreError.wrongCount`.
    public init(values: [Float], scale: CommandScale) throws {
        guard values.count == Command.count else {
            throw UEMACoreError.wrongCount(type: "Command", expected: Command.count, found: values.count)
        }
        self.values = values
        self.scale = scale
    }

    private init(unchecked values: [Float], scale: CommandScale) {
        self.values = values
        self.scale = scale
    }

    /// A command whose 234 numbers all equal `value`.
    public init(repeating value: Float, scale: CommandScale) {
        self.init(unchecked: Array(repeating: value, count: Command.count), scale: scale)
    }

    /// The all-zero command at `scale`.
    public static func zero(scale: CommandScale) -> Command {
        Command(repeating: 0, scale: scale)
    }

    /// The flat index of one number: handle row x 9 + column.
    public static func index(handle: BodyHandle, column: CommandColumn) -> Int {
        handle.rawValue * Command.numbersPerHandle + column.rawValue
    }

    /// One number of the command, by handle and column.
    public subscript(handle: BodyHandle, column: CommandColumn) -> Float {
        get { values[Command.index(handle: handle, column: column)] }
        set { values[Command.index(handle: handle, column: column)] = newValue }
    }

    /// The 9 numbers of one handle's row.
    public func row(_ handle: BodyHandle) -> [Float] {
        let start = handle.rawValue * Command.numbersPerHandle
        return Array(values[start ..< start + Command.numbersPerHandle])
    }

    /// Replaces one handle's 9 numbers; a row of another count throws
    /// `UEMACoreError.wrongCount`.
    public mutating func setRow(_ handle: BodyHandle, to row: [Float]) throws {
        guard row.count == Command.numbersPerHandle else {
            throw UEMACoreError.wrongCount(
                type: "Command row", expected: Command.numbersPerHandle, found: row.count)
        }
        let start = handle.rawValue * Command.numbersPerHandle
        values.replaceSubrange(start ..< start + Command.numbersPerHandle, with: row)
    }

    /// The target position x y z of one handle, at this command's scale.
    public func position(of handle: BodyHandle) -> [Float] {
        Array(row(handle)[0 ..< 3])
    }

    /// The target orientation quaternion x y z w of one handle.
    public func quaternion(of handle: BodyHandle) -> [Float] {
        Array(row(handle)[3 ..< 7])
    }

    /// The radius of one handle (column 7).
    public func radius(of handle: BodyHandle) -> Float { self[handle, .radius] }

    /// The strength of one handle (column 8), 0 to 1 once clipped by the decoder.
    public func strength(of handle: BodyHandle) -> Float { self[handle, .strength] }

    /// The command at `.normalized` scale: every position column divided by 200
    /// (`normalize_executed_command`, encoder_registry.py:245-267); the other columns are
    /// copied. A command already at `.normalized` scale is returned unchanged.
    public func normalized() -> Command {
        guard scale == .centimetres else { return self }
        var out = values
        for handle in BodyHandle.allCases {
            for column in CommandColumn.position {
                let index = Command.index(handle: handle, column: column)
                out[index] = out[index] / Command.positionRangeCentimetres
            }
        }
        return Command(unchecked: out, scale: .normalized)
    }

    /// The gaze azimuth and elevation (radians) read from the eyes' row as
    /// `_gaze_from_handle` does (lem_model.py:5122-5140): the quaternion x y z w of row 23
    /// divided by its norm (floor 1e-6); yaw = atan2(2 (w z + x y), 1 - 2 (y y + z z));
    /// pitch = asin(clamp(2 (w y - z x), -1, 1)); azimuth = yaw clamped to -0.785 ... 0.785;
    /// elevation = pitch (eye_row_convention 'legacy') clamped to -0.820 ... 0.576.
    public var gazeAngles: GazeAngles {
        let q = quaternion(of: .gaze)
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).squareRoot()
        let divisor = Double(max(norm, 1e-6))
        let x = Double(q[0]) / divisor
        let y = Double(q[1]) / divisor
        let z = Double(q[2]) / divisor
        let w = Double(q[3]) / divisor
        let yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        let sine = min(max(2 * (w * y - z * x), -1), 1)
        let pitch = asin(sine)
        return GazeAngles(azimuth: yaw, elevation: pitch).clamped()
    }

    /// Writes gaze angles into the eyes' row as `gaze_to_handle` does (lem_model.py:5094-5107,
    /// convention 'legacy': pitch = elevation): with ca = cos(azimuth / 2), sa = sin(azimuth / 2),
    /// ce = cos(pitch / 2), se = sin(pitch / 2) the quaternion becomes x = -sa se, y = ca se,
    /// z = sa ce, w = ca ce. The row's position, radius and strength are left as they are.
    public mutating func setGaze(_ angles: GazeAngles) {
        let ca = cos(angles.azimuth / 2)
        let sa = sin(angles.azimuth / 2)
        let ce = cos(angles.elevation / 2)
        let se = sin(angles.elevation / 2)
        self[.gaze, .quaternionX] = Float(-sa * se)
        self[.gaze, .quaternionY] = Float(ca * se)
        self[.gaze, .quaternionZ] = Float(sa * ce)
        self[.gaze, .quaternionW] = Float(ca * ce)
    }

    private enum CodingKeys: String, CodingKey {
        case scale
        case values
    }

    /// Decodes through the count check, so a decoded command also holds 234 numbers.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            values: container.decode([Float].self, forKey: .values),
            scale: container.decode(CommandScale.self, forKey: .scale))
    }
}

// MARK: - Gaze angles

/// The eyes' azimuth and elevation in radians: the pair the eye servo takes
/// (command_decoder.md section 1, `gaze`). The real eye joint ranges are azimuth -0.785 to
/// 0.785 (plus or minus 45 degrees) and elevation -0.820 to 0.576 (-47 to +33 degrees)
/// (gaze_reflex.py:41-42).
public struct GazeAngles: Equatable, Hashable, Codable, Sendable {
    /// `AZ_RANGE` (-0.785, 0.785) rad (gaze_reflex.py:41).
    public static let azimuthRange: ClosedRange<Double> = (-0.785) ... (0.785)

    /// `EL_RANGE` (-0.820, 0.576) rad (gaze_reflex.py:42).
    public static let elevationRange: ClosedRange<Double> = (-0.820) ... (0.576)

    /// Azimuth (yaw) in radians, positive to the left as the Python yaw formula gives it.
    public var azimuth: Double

    /// Elevation (pitch) in radians.
    public var elevation: Double

    /// Angles as given, not clamped; `clamped()` applies the two ranges.
    public init(azimuth: Double, elevation: Double) {
        self.azimuth = azimuth
        self.elevation = elevation
    }

    /// Both angles clamped to the real eye joint ranges.
    public func clamped() -> GazeAngles {
        let az = min(max(azimuth, GazeAngles.azimuthRange.lowerBound), GazeAngles.azimuthRange.upperBound)
        let el = min(max(elevation, GazeAngles.elevationRange.lowerBound), GazeAngles.elevationRange.upperBound)
        return GazeAngles(azimuth: az, elevation: el)
    }

    /// Azimuth 0, elevation 0.
    public static let straightAhead = GazeAngles(azimuth: 0, elevation: 0)
}

// MARK: - Handle mask

/// The decoder's `handle_mask`: 26 booleans, one per handle, true where the handle's row is
/// used (command_decoder.md section 1). Every curriculum phase of the released run has all 26
/// true (lem_model.py:5720-5721, read November 4, 2026), and the achieved motor row carries it as 1.0 or 0.0
/// (proprio_rows.md section 5 step 1).
public struct HandleMask: Equatable, Hashable, Codable, Sendable {
    /// The entries per mask: 26.
    public static let count = BodyHandle.count

    /// One boolean per handle, by row.
    public let values: [Bool]

    /// Wraps exactly 26 booleans; any other count throws `UEMACoreError.wrongCount`.
    public init(values: [Bool]) throws {
        guard values.count == HandleMask.count else {
            throw UEMACoreError.wrongCount(type: "HandleMask", expected: HandleMask.count, found: values.count)
        }
        self.values = values
    }

    /// A mask whose 26 entries all equal `value`.
    public init(repeating value: Bool) {
        values = Array(repeating: value, count: HandleMask.count)
    }

    /// The released run's mask: all 26 true.
    public static let allTrue = HandleMask(repeating: true)

    /// The entry of one handle.
    public subscript(handle: BodyHandle) -> Bool { values[handle.rawValue] }

    /// The 26 entries as 1.0 (true) or 0.0 (false), the achieved motor row's column 9.
    public var asFloats: [Float] { values.map { $0 ? 1 : 0 } }

    private enum CodingKeys: String, CodingKey {
        case values
    }

    /// Decodes through the count check.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(values: container.decode([Bool].self, forKey: .values))
    }
}

// MARK: - Finger joint goals

/// The decoder's `finger_goals`: 44 goal angles in radians for the 44 hand hinge joints, in
/// the geometry sequence of the receipt's `hand_response.joint_names` (command_decoder.md
/// section 1). The planner's hand descent produces them every 10th model step
/// (specification REQ-B-12).
public struct FingerJointGoals: Equatable, Hashable, Codable, Sendable {
    /// The hand hinge joints: 44.
    public static let count = 44

    /// The 44 goal angles in radians.
    public let angles: [Double]

    /// Wraps exactly 44 angles; any other count throws `UEMACoreError.wrongCount`.
    public init(angles: [Double]) throws {
        guard angles.count == FingerJointGoals.count else {
            throw UEMACoreError.wrongCount(
                type: "FingerJointGoals", expected: FingerJointGoals.count, found: angles.count)
        }
        self.angles = angles
    }

    private enum CodingKeys: String, CodingKey {
        case angles
    }

    /// Decodes through the count check.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(angles: container.decode([Double].self, forKey: .angles))
    }
}

// MARK: - Motor controls

/// The 90 motor controls the decoder writes to the body each env step (`ctrl`, [nu] float64,
/// nu = 90 on the released crib scene, `PhysicsSizes.released`), each within the actuator
/// control range -1 to 1 (plan row WI-25: "writes ctrl clipped to -1..1").
public struct MotorControls: Equatable, Hashable, Codable, Sendable {
    /// The motors of the released body: 90 (`PhysicsSizes.released.nu`).
    public static let count = 90

    /// The control range of every motor: -1 to 1 (`actuator_ctrlrange`; plan row WI-25).
    public static let range: ClosedRange<Double> = (-1.0) ... (1.0)

    /// The 90 controls, in MuJoCo actuator sequence.
    public let values: [Double]

    /// Wraps exactly 90 controls; any other count throws `UEMACoreError.wrongCount`.
    public init(values: [Double]) throws {
        guard values.count == MotorControls.count else {
            throw UEMACoreError.wrongCount(
                type: "MotorControls", expected: MotorControls.count, found: values.count)
        }
        self.values = values
    }

    private init(unchecked values: [Double]) {
        self.values = values
    }

    /// All 90 controls at 0.
    public static let zero = MotorControls(unchecked: Array(repeating: 0, count: MotorControls.count))

    /// True when every control lies within -1 to 1.
    public var isWithinRange: Bool {
        values.allSatisfy { MotorControls.range.contains($0) }
    }

    /// Every control clipped to -1 to 1 (command_decoder.md section 3 step 5).
    public func clipped() -> MotorControls {
        MotorControls(unchecked: values.map {
            min(max($0, MotorControls.range.lowerBound), MotorControls.range.upperBound)
        })
    }

    private enum CodingKeys: String, CodingKey {
        case values
    }

    /// Decodes through the count check.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(values: container.decode([Double].self, forKey: .values))
    }
}
