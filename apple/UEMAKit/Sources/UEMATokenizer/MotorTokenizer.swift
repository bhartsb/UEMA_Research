// MotorTokenizer.swift: the motor tokenizer of the UEMA Apple port (work item WI-30, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-4; contract
// apple/UEMAKit/Specs/proprio_rows.md section 5). Nothing in this file was compiled on the
// machine that wrote it. The "First build on a Mac" checklist of this module is in
// VisionPatcher.swift's header.
//
// INPUT. The motor reader's achieved rows (`motor_to_echo`, Source/Python/mimo_bridge/
// datasource.py:173-192), float32 [26, 10], one row per body handle: columns 0-2 the achieved
// position in cm, world frame; 3-6 the achieved quaternion x y z w; 7 the commanded radius;
// 8 the commanded strength; 9 the handle mask as 1.0 or 0.0. For the context row of an earlier
// model step, also the executed command of that model step (`Command`, 26 x 9 numbers).
//
// OPERATION.
//   Achieved half: columns 0-2 / 200 cm (`handles[:, :3] /= cfg.position_range_cm`,
//   datasource.py:150; `position_range_cm` 200.0, tokenizer/config.py:263; motor.py:75-77 for
//   the 24-row path); columns 3-9 copied.
//   Widening to 19 columns (`MOTOR_ACHIEVED_F` 10, `MOTOR_COMMAND_F` 9, `MOTOR_INPUT_F` 19,
//   lem_model.py:69-71): columns 10-18 hold the executed command with its positions / 200
//   (`command_for_encoder`, lem_model.py:139-149 -> `normalize_executed_command`,
//   encoder_registry.py:245-267). The LIVE model step's command columns are zeros, since its
//   command has not been chosen yet (proprio_rows.md section 5 step 3, lem_model.py:2657-2661);
//   `tokenize` gives that row. The PREVIOUS model step's slot in the context buffer is rewritten
//   with its achieved half and its executed command (lem_model.py:2640-2656);
//   `encoderInput(achieved:executedCommand:)` gives that row.
//   Energy and capability (motor.py:85-108; proprio_rows.md section 5 step 4) are not encoder
//   inputs: `MotorEnergyScaler` divides them by 1,000 (`energy_max`, tokenizer/config.py:273)
//   for the wellness capability term.
//
// OUTPUT. float32 [26, 19], the motor encoder's input; `MotorEnergyScaler` gives energy [3] and
// capability [21].

import Foundation
import UEMACore

/// The motor tokenizer: the achieved handle rows widened with the executed command into the
/// motor encoder's [26, 19] input (lem_model.py:69-71, :2640-2661).
public struct MotorTokenizer: Tokenizer, Sendable {
    /// Centimetres per unit of the position columns: 200 (tokenizer/config.py:263;
    /// `COMMAND_POSITION_RANGE_CM`, encoder_registry.py:242).
    public static let positionRangeCentimetres: Float = 200

    /// Achieved columns: 10 (`MOTOR_ACHIEVED_F`, lem_model.py:69).
    public static let achievedColumns = 10

    /// Command columns: 9 (`MOTOR_COMMAND_F`, lem_model.py:70).
    public static let commandColumns = Command.numbersPerHandle

    /// Columns of the encoder input: 19 (`MOTOR_INPUT_F`, lem_model.py:71).
    public static let inputColumns = achievedColumns + commandColumns

    /// Rows: 26 handles.
    public static let rowCount = Command.handleCount

    /// The sense served: motor.
    public var sense: SenseName { .motor }

    /// A tokenizer; it holds no state.
    public init() {}

    /// INPUT: `achieved` [26, 10] as this file's header lists, positions in cm; `executedCommand`
    /// the command in force over the model step, or nil for the live model step. OUTPUT: [26, 19]:
    /// achieved positions / 200, columns 3-9 copied, then the command's 9 numbers with positions
    /// / 200 (`Command.normalized()`), or 9 zeros when nil.
    public func encoderInput(achieved: ShapedArray, executedCommand: Command?) throws -> ShapedArray {
        let count = try TokenizerChecks.rows(
            of: achieved, columns: [MotorTokenizer.achievedColumns], sense: .motor, name: "primary",
            expected: "[26, 10]")
        guard count == MotorTokenizer.rowCount else {
            throw TokenizerError.wrongShape(sense: .motor, array: "primary", expected: "[26, 10]", found: achieved.shape)
        }
        let commandValues = executedCommand?.normalized().values
        let a = achieved.values
        let width = MotorTokenizer.inputColumns
        var out = [Float](repeating: 0, count: MotorTokenizer.rowCount * width)
        for i in 0 ..< MotorTokenizer.rowCount {
            let s = i * MotorTokenizer.achievedColumns
            let o = i * width
            for j in 0 ..< MotorTokenizer.achievedColumns {
                out[o + j] = j < 3 ? a[s + j] / MotorTokenizer.positionRangeCentimetres : a[s + j]
            }
            if let command = commandValues {
                let c = i * MotorTokenizer.commandColumns
                for j in 0 ..< MotorTokenizer.commandColumns {
                    out[o + MotorTokenizer.achievedColumns + j] = command[c + j]
                }
            }
        }
        return try ShapedArray(shape: [MotorTokenizer.rowCount, width], values: out)
    }

    /// The live model step's encoder input [26, 19] (command columns zero); `isFresh` carried.
    public func tokenize(_ reading: SenseReading) throws -> EncoderInput {
        try TokenizerChecks.requireSense(reading, .motor)
        return EncoderInput(
            sense: .motor,
            primary: try encoderInput(achieved: reading.primary, executedCommand: nil),
            isFresh: reading.isFresh)
    }
}

/// The motor energy and capability numbers divided by the energy scale (tokenize_motor,
/// Source/Python/tokenizer/modalities/motor.py:85-108): inputs of the wellness capability term
/// (proprio_rows.md section 5 step 4), not of the motor encoder.
public struct MotorEnergyScaler: Equatable, Sendable {
    /// Energy units per output unit: 1,000 (`energy_max`, tokenizer/config.py:273).
    public static let energyMax: Double = 1000

    /// Body sets: 10 (`NUM_BODY_SETS`; proprio_rows.md section 2, `body_sets` [10, 2]).
    public static let bodySetCount = 10

    /// A scaler; it holds no state.
    public init() {}

    /// INPUT: the three energies of `ProprioAdapter.read` (proprio_rows.md section 2:
    /// ph_work = sum(effort) x dt, pcc_hold = sum |actuator force| x dt x 0.01, total = their
    /// sum). OUTPUT: [ph_work, pcc_hold, total] / 1000 as float32 (motor.py:85-95; the Python
    /// divides the float32 value in float64 and stores float32).
    public func energy(physicalWork: Float, holding: Float, total: Float) -> [Float] {
        [physicalWork, holding, total].map { Float(Double($0) / MotorEnergyScaler.energyMax) }
    }

    /// INPUT: the 10 body-set capacities (0 to 1), the 10 accumulated efforts and the total
    /// energy. OUTPUT: [21] = [capacity 0..9, accumulated 0..9 / 1000, total / 1000]
    /// (motor.py:97-108). Throws `TokenizerError.wrongStatisticsCount` for lists of other than 10.
    public func capability(capacities: [Float], accumulated: [Float], totalEnergy: Float) throws -> [Float] {
        guard capacities.count == MotorEnergyScaler.bodySetCount else {
            throw TokenizerError.wrongStatisticsCount(
                name: "capacities", expected: MotorEnergyScaler.bodySetCount, found: capacities.count)
        }
        guard accumulated.count == MotorEnergyScaler.bodySetCount else {
            throw TokenizerError.wrongStatisticsCount(
                name: "accumulated", expected: MotorEnergyScaler.bodySetCount, found: accumulated.count)
        }
        var out = capacities
        out.append(contentsOf: accumulated.map { Float(Double($0) / MotorEnergyScaler.energyMax) })
        out.append(Float(Double(totalEnergy) / MotorEnergyScaler.energyMax))
        return out
    }

    /// Numbers in the motor reader's auxiliary array: 24 = energy 3 + capability 21
    /// (UEMASenses/MotorReader.swift header, "Auxiliary, float32 [24]").
    public static let readingAuxiliaryCount = 24

    /// INPUT: the motor reader's auxiliary array [24] in physical units: (ph_work, pcc_hold,
    /// total), then (capacity 0..9, accumulated 0..9, total) (UEMASenses/MotorReader.swift).
    /// OUTPUT: `energy` [3] and `capability` [21] divided as `energy(...)` and
    /// `capability(...)` divide them (motor.py:85-108). Throws `TokenizerError.wrongShape` for
    /// another shape.
    public func scaled(readingAuxiliary auxiliary: ShapedArray) throws -> (energy: [Float], capability: [Float]) {
        guard auxiliary.shape == [MotorEnergyScaler.readingAuxiliaryCount] else {
            throw TokenizerError.wrongShape(
                sense: .motor, array: "auxiliary", expected: "[24]: energy 3 then capability 21", found: auxiliary.shape)
        }
        let v = auxiliary.values
        let sets = MotorEnergyScaler.bodySetCount
        let energy = self.energy(physicalWork: v[0], holding: v[1], total: v[2])
        let capability = try self.capability(
            capacities: Array(v[3 ..< 3 + sets]), accumulated: Array(v[3 + sets ..< 3 + 2 * sets]),
            totalEnergy: v[3 + 2 * sets])
        return (energy, capability)
    }
}
