// MotorReader.swift: the motor sense of the UEMA Apple port, its achieved half (work item
// WI-27, target B of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification
// REQ-B-4; contract apple/UEMAKit/Specs/proprio_rows.md section 5).
//
// WHAT THIS FILE IS. The port of `motor_to_echo` (Source/Python/mimo_bridge/datasource.py:
// 173-192) and of the energy and capability arrays the motor tokenizer reads
// (tokenizer/modalities/motor.py:85-108), over the env step's `ProprioceptionSample`
// (ProprioceptionReader.swift) and the command executed in that env step.
//
// INPUT. The shared `ProprioceptionSampling` (its bones float32 [26, 11], energy [3],
// capacities [10], accumulated effort [10]) and an `ExecutedCommandProviding`: the command
// the decoder executed in the last env step (scale `.centimetres`; only its radius and
// strength columns are read, both dimensionless) and its handle mask (datasource.py:285 and
// :420 call `motor_to_echo(motor, mask, bones=raw.proprio.bones)` with the executed command
// of that env step).
//
// OPERATION (datasource.py:182-192), per row i of 26:
//   echo[i, 0:3] = bones[i, 4:7]   achieved position, cm, world frame
//   echo[i, 3:7] = bones[i, 0:4]   achieved quaternion x y z w (the gaze row head-relative)
//   echo[i, 7]   = command[i, 7]   commanded radius (0 when no command was executed yet)
//   echo[i, 8]   = command[i, 8]   commanded strength (0 when no command was executed yet)
//   echo[i, 9]   = mask[i] as 1.0 / 0.0, 1.0 when there is no mask
// Auxiliary, float32 [24]: energy [3] = (ph_work, pcc_hold, total) of the env step
// (proprio_adapter.py:239-241), then capability [21] = (cap_0 ... cap_9, accum_0 ... accum_9,
// total) by `BodySet` ordinal (motor.py:97-108 before its divisions).
//
// OUTPUT. `SenseReading(sense: .motor, primary: [26, 10], auxiliary: [24])` in physical
// units. The tokenizer (work item WI-30) divides: echo columns 0-2 by 200
// (`position_range_cm`, tokenizer/config.py:263; datasource.py:151 `handles[:, :3] /= 200`);
// energy 0-2, accum (auxiliary 13-22) and total (auxiliary 23) by 1000 (`energy_max`,
// config.py:273; motor.py:91-95, :105-107). It then widens the 10 achieved numbers with 9
// command numbers (lem_model.py:2640-2661): zeros on the live model step, the executed
// command's positions over 200 on the previous model step's slot.
//
// SOLID. Single responsibility: this file turns a sample and the executed command into the
// motor reading. Dependency inversion: the reader takes the two protocols at init;
// `ExecutedCommandRecord` is one concrete provider the session (WI-32) may inject.
//
// SOURCES READ BEFORE WRITING (November 4, 2026): Source/Python/mimo_bridge/datasource.py
// lines 100-192 (motor_to_echo, _payload_to_frame), 270-295 and 410-425 (the two callers);
// Source/Python/tokenizer/modalities/motor.py lines 75-114; tokenizer/config.py lines 263,
// 273; apple/UEMAKit/Specs/proprio_rows.md section 5; apple/UEMAKit/Specs/
// motor_controller_input.md section 1 (`flat_handles` = this achieved half, 26 x 10 = 260).
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain). The "First build on a Mac" checklist of this work item is in
// ProprioceptionReader.swift's header.

import Foundation
import UEMACore

// MARK: - The executed command

/// The command the decoder executed in the last env step and its handle mask, as the motor
/// reader needs them. nil before the first executed command.
public protocol ExecutedCommandProviding: AnyObject {
    /// The command executed in the last env step (any scale; columns 7 and 8 are read).
    var executedCommand: Command? { get }

    /// The handle mask that command was executed with; nil means no mask (every row 1.0).
    var executedMask: HandleMask? { get }
}

/// A holder the session writes the executed command into after each decode; the motor reader
/// reads it through `ExecutedCommandProviding`.
public final class ExecutedCommandRecord: ExecutedCommandProviding {
    /// The last recorded command; nil before the first `record(_:mask:)` or after `clear()`.
    public private(set) var executedCommand: Command?

    /// The mask recorded with that command; nil means no mask (every row 1.0).
    public private(set) var executedMask: HandleMask?

    /// An empty record.
    public init() {}

    /// Records the command executed in this env step and its mask.
    public func record(_ command: Command, mask: HandleMask?) {
        executedCommand = command
        executedMask = mask
    }

    /// Forgets the recorded command.
    public func clear() {
        executedCommand = nil
        executedMask = nil
    }
}

// MARK: - The reader

/// The motor `SenseReader` (specification REQ-B-4): the achieved rows float32 [26, 10] and
/// the energy and capability float32 [24] of the last env step.
public final class MotorReader: SenseReader {
    /// Rows: 26.
    public static let rowCount = ProprioceptionLayout.rowCount

    /// Numbers per achieved row: 10 (`MOTOR_ACHIEVED_F`, lem_model.py:69).
    public static let achievedColumns = 10

    /// Numbers in the energy block: 3.
    public static let energyCount = 3

    /// Numbers in the capability block: 21 (motor.py:98).
    public static let capabilityCount = 21

    /// Numbers in the auxiliary array: 3 + 21 = 24.
    public static let auxiliaryCount = energyCount + capabilityCount

    /// Auxiliary indices the tokenizer divides by 1000 (`energy_max`): the energy block 0-2,
    /// the accumulated effort 13-22 and the total energy 23.
    public static let auxiliaryIndicesOverEnergyMax: [Int] = Array(0 ..< 3) + Array(13 ..< 24)

    public let sense: SenseName = .motor

    private let tracker: ProprioceptionSampling
    private let executed: ExecutedCommandProviding

    /// A reader over the shared tracker and the executed-command provider.
    public init(tracker: ProprioceptionSampling, executed: ExecutedCommandProviding) {
        self.tracker = tracker
        self.executed = executed
    }

    /// Advances the shared fatigue model once for this env step.
    public func observeEnvStep(_ physics: any PhysicsReading) throws {
        _ = try tracker.sample(physics)
    }

    public func read(_ physics: any PhysicsReading, at step: ModelStep) throws -> SenseReading {
        let sample = try tracker.sample(physics)
        let rows = try MotorReader.achievedRows(
            from: sample, command: executed.executedCommand, mask: executed.executedMask)
        let auxiliary = try MotorReader.energyAndCapability(from: sample)
        return SenseReading(sense: .motor, primary: rows, auxiliary: auxiliary)
    }

    public func reset() {
        tracker.reset()
    }

    /// The [26, 10] achieved rows (`motor_to_echo`, datasource.py:182-192).
    public static func achievedRows(from sample: ProprioceptionSample, command: Command?, mask: HandleMask?) throws -> ShapedArray {
        var values = Array(repeating: Float(0), count: rowCount * achievedColumns)
        let maskValues = mask?.asFloats
        for row in 0 ..< rowCount {
            let base = row * achievedColumns
            for axis in 0 ..< 3 {
                values[base + axis] = sample.bone(row: row, column: 4 + axis)
            }
            for component in 0 ..< 4 {
                values[base + 3 + component] = sample.bone(row: row, column: component)
            }
            if let command = command, let handle = BodyHandle(rawValue: row) {
                values[base + 7] = command[handle, .radius]
                values[base + 8] = command[handle, .strength]
            }
            values[base + 9] = maskValues?[row] ?? 1
        }
        return try ShapedArray(shape: [rowCount, achievedColumns], values: values)
    }

    /// The [24] auxiliary array: energy (3) then capability (21) = capacities (10),
    /// accumulated effort (10), total energy (1), all before the tokenizer's divisions.
    public static func energyAndCapability(from sample: ProprioceptionSample) throws -> ShapedArray {
        var values: [Float] = sample.energy
        values.append(contentsOf: sample.capacities.map { Float($0) })
        values.append(contentsOf: sample.accumulated.map { Float($0) })
        values.append(sample.energy[2])
        return try ShapedArray(shape: [auxiliaryCount], values: values)
    }
}
