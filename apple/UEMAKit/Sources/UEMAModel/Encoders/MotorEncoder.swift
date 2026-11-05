// MotorEncoder.swift: the motor sense's encoder of the released model in MLX Swift (work item WI-35
// of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-9). Nothing in
// this file was compiled on the machine that wrote it (no Swift toolchain on Windows).
//
// INPUT: 26 body handle rows x 19 numbers [26, 19] (UEMATokenizer's MotorTokenizer, WI-30;
//   apple/UEMAKit/Specs/proprio_rows.md section 5): the achieved handle state (10 numbers,
//   positions / 200 cm) then the executed command (9 numbers, positions / 200 cm). On the live pass
//   the 9 command columns are zeros (the fixture's input/motor [1, 26, 19]); the context back-fill
//   (ForwardModel.swift) and the planner (WI-39) fill them with a command.
// OPERATION: the cross-patch attention encoder of TactileEncoder.swift (cross_patch.py:24-127) with
//   input width 19 (lem/master_config.yaml:189) and the per-handle position vectors patch_pos
//   [1, 26, 192] added after patch_mlp (cross_patch.py:100-102; master_config.yaml:199
//   patch_pos_embedding true), 2 layers of 4 heads, a 4-head pool, no biases, and the liveness
//   repair's all-element path added to the pool's output (TactileEncoder.swift step 6; weights
//   liveness_repair.paths.motor.*, up.weight [192, 494]).
//   The released run also had the motor asked-change path on (args.json motor_asked_change on;
//   lem/model/motor_asked_change.py, component motor_asked_change, 131,728 parameters, outside the
//   inference set of the manifest): a forward hook adds s x u to this encoder's pooled vector, with
//   s its standard deviation over the 192 numbers and u a function of the asked change (command
//   minus achieved pose). For a row without a command (the live pass's zero command columns) its
//   output is exactly 0 (motor_asked_change.py, "WHEN A HANDLE'S ROW ASKS SOMETHING"), so the live
//   token equals this encoder's; a row with a command differs by the path's output. That path is
//   the PooledVectorAdjusting hook below, injected; it is not ported in this work item.
// OUTPUT: [B, 1, 192] before the shared output LayerNorm (the fixture's encoder_output/motor);
//   backbone(...) gives the per-handle features [B, 26, 192] (encoder_features/motor).
// PARAMETERS: 1,079,808 (manifest component enc_motor; 40,512 + 4,992 + 2 x 443,136 + 148,032).
// GRADIENT: callAsFunction(_:) takes an MLXArray, so the planner (WI-39) differentiates the motor
//   token with respect to the command columns through MLX's grad transforms.
// FIRST BUILD ON A MAC: see TransformerCore.swift, steps 1 to 4.

import Foundation
import MLX
import UEMACore

/// A change applied to the motor encoder's pooled vector given its input rows: the place where the
/// released run's asked-change path (lem/model/motor_asked_change.py attach(): v -> v + s x u)
/// enters. Implementations return `pooled` unchanged for rows whose command columns are zeros.
public protocol PooledVectorAdjusting {
    /// The adjusted pooled vector [B, 1, 192] for `pooled` [B, 1, 192] and `rows` [B, 26, 19].
    func adjust(pooled: MLXArray, rows: MLXArray) -> MLXArray
}

/// The motor encoder: the cross-patch attention encoder over 26 handle rows x 19 numbers with
/// per-handle position vectors.
public struct MotorEncoder: SenseEncoding {
    /// Body handle rows: 26.
    public static let handles = BodyHandle.count
    /// Numbers per row: 10 achieved + 9 command = 19.
    public static let numbersPerRow = 19
    /// Achieved numbers at the front of a row: 10.
    public static let achievedNumbers = 10
    /// Parameters: 1,079,808.
    public static let parameterCount = 1_079_808

    public let sense: SenseName = .motor
    /// The shared encoder body.
    public let core: CrossPatchAttentionEncoder
    /// The asked-change hook; nil adds nothing.
    public let adjustment: (any PooledVectorAdjusting)?

    /// The encoder stored under `source` (the released file seen through
    /// `scoped("encoder_registry.encoders.motor.")`), with the liveness repair's pool path when given
    /// (AllElementPath.released(sense: .motor, ...)) and an optional pooled-vector hook.
    public init(source: any MLXTensorSource, poolPath: AllElementPath? = nil,
                adjustment: (any PooledVectorAdjusting)? = nil) throws {
        core = try CrossPatchAttentionEncoder(source: source, inputWidth: MotorEncoder.numbersPerRow,
                                              positionSlots: MotorEncoder.handles, poolPath: poolPath)
        self.adjustment = adjustment
    }

    public var parameterCount: Int { core.parameterCount }

    /// The pooled vector [B, 1, 192] of rows [B, 26, 19].
    public func callAsFunction(_ rows: MLXArray) -> MLXArray {
        let pooled = core(rows)
        guard let adjustment else {
            return pooled
        }
        return adjustment.adjust(pooled: pooled, rows: rows)
    }

    /// The encoder input rows [B, 26, 19] of achieved states [B, 26, >= 10] (the first 10 numbers
    /// used) and a command [B, 26, 9] at position_range scale (positions / 200 cm):
    /// build_motor_input of encoder_registry.py:271-299 with the 19-wide encoder.
    public static func rows(achieved: MLXArray, command: MLXArray) -> MLXArray {
        concatenated([achieved[0..., 0..., 0 ..< achievedNumbers], command], axis: -1)
    }

    public func encode(_ input: EncoderInput) throws -> MLXArray {
        guard input.sense == .motor else {
            throw ModelPortError.missingInput(.motor)
        }
        let rows = try NeuralOps.batched(input.primary, expected: [MotorEncoder.handles, MotorEncoder.numbersPerRow],
                                         what: "motor rows")
        return self(rows)
    }
}
