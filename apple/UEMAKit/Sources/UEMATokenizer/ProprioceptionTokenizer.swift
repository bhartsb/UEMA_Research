// ProprioceptionTokenizer.swift: the proprioception tokenizer of the UEMA Apple port (work item
// WI-30, target B of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification
// REQ-B-4; contract apple/UEMAKit/Specs/proprio_rows.md section 4). Nothing in this file was
// compiled on the machine that wrote it. The "First build on a Mac" checklist of this module is
// in VisionPatcher.swift's header.
//
// INPUT. The proprioception reader's rows (proprio_rows.md section 2; `ProprioAdapter.read`,
// Source/Python/mimo_bridge/proprio_adapter.py:201-243), float32 [26, 11] (or [26, 12], whose
// column 11 is ignored and rewritten), one row per body handle in the command's row sequence:
// columns 0-3 orientation quaternion x y z w (world frame; the gaze row head-relative);
// 4-6 position in cm, world frame; 7-9 velocity in cm/s, world frame; 10 the fatigue capacity
// of the row's body set, 0 to 1.
//
// OPERATION (Source/Python/mimo_bridge/datasource.py:127-141, the 26-row path; the same rules
// as tokenizer/modalities/proprioception.py:66-83 for 24 rows), in float32:
//   columns 4-6: (position - position of row 0, wrist_L) / 200 cm (`position_range_cm` 200.0,
//                tokenizer/config.py:263; proprioception.py:72);
//   columns 7-9: velocity / 500 cm/s (`velocity_range_cms` 500.0, config.py:264;
//                proprioception.py:75);
//   column 11:   row index / 25 (`proprio_index_range` 25.0, config.py:271; proprioception.py:82;
//                datasource.py:141 divides in float64 and stores float32);
//   columns 0-3 and 10 are copied.
//
// OUTPUT. float32 [312] = the [26, 12] rows flattened row-major (`reshape(-1)`,
// lem_model.py:2413), the proprioception encoder's input.

import Foundation
import UEMACore

/// The proprioception tokenizer: 26 handle rows into the 312 inputs of the proprioception
/// encoder (datasource.py:127-141).
public struct ProprioceptionTokenizer: Tokenizer, Sendable {
    /// Centimetres per unit of the position columns: 200 (tokenizer/config.py:263).
    public static let positionRangeCentimetres: Float = 200

    /// Centimetres per second per unit of the velocity columns: 500 (tokenizer/config.py:264).
    public static let velocityRangeCentimetresPerSecond: Float = 500

    /// Rows per unit of the row-index column: 25 (tokenizer/config.py:271).
    public static let indexRange: Double = 25

    /// Rows per reading: 26 (body layout trunk_v1, body_layout.py:40-55).
    public static let rowCount = Command.handleCount

    /// Numbers per output row: 12.
    public static let columnCount = 12

    /// The sense served: proprioception.
    public var sense: SenseName { .proprioception }

    /// A tokenizer; it holds no state.
    public init() {}

    /// INPUT: `rows` [26][11] or [26][12] as this file's header lists. OUTPUT: [26][12].
    public static func normalisedRows(_ rows: ShapedArray) throws -> [Float] {
        let count = try TokenizerChecks.rows(
            of: rows, columns: [11, 12], sense: .proprioception, name: "primary",
            expected: "[26, 11] or [26, 12]")
        guard count == rowCount else {
            throw TokenizerError.wrongShape(
                sense: .proprioception, array: "primary", expected: "[26, 11] or [26, 12]", found: rows.shape)
        }
        let width = rows.shape[1]
        let v = rows.values
        let rootX = v[4]
        let rootY = v[5]
        let rootZ = v[6]
        var out = [Float](repeating: 0, count: rowCount * columnCount)
        for i in 0 ..< rowCount {
            let s = i * width
            let o = i * columnCount
            out[o] = v[s]
            out[o + 1] = v[s + 1]
            out[o + 2] = v[s + 2]
            out[o + 3] = v[s + 3]
            out[o + 4] = (v[s + 4] - rootX) / positionRangeCentimetres
            out[o + 5] = (v[s + 5] - rootY) / positionRangeCentimetres
            out[o + 6] = (v[s + 6] - rootZ) / positionRangeCentimetres
            out[o + 7] = v[s + 7] / velocityRangeCentimetresPerSecond
            out[o + 8] = v[s + 8] / velocityRangeCentimetresPerSecond
            out[o + 9] = v[s + 9] / velocityRangeCentimetresPerSecond
            out[o + 10] = v[s + 10]
            out[o + 11] = Float(Double(i) / indexRange)
        }
        return out
    }

    /// The encoder input [312] of one reading; `isFresh` carried.
    public func tokenize(_ reading: SenseReading) throws -> EncoderInput {
        try TokenizerChecks.requireSense(reading, .proprioception)
        let values = try ProprioceptionTokenizer.normalisedRows(reading.primary)
        return EncoderInput(
            sense: .proprioception,
            primary: try ShapedArray(shape: [values.count], values: values),
            isFresh: reading.isFresh)
    }
}
