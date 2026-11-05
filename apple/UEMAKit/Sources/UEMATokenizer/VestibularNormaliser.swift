// VestibularNormaliser.swift: the vestibular tokenizer of the UEMA Apple port (work item WI-30,
// target B of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-4;
// contract apple/UEMAKit/Specs/vestibular_fields.md sections 2 and 3). Nothing in this file was
// compiled on the machine that wrote it. The "First build on a Mac" checklist of this module is
// in VisionPatcher.swift's header.
//
// INPUT. The vestibular reader's 29 fields of the model step's last env step, float32 [29]
// (`VESTIBULAR_FIELDS`, vestibular_adapter.py:40-53; vestibular_fields.md section 1): head
// acceleration in g (3), head rotation rate / 2 rad/s (3), neck angles on -1..1 (3), neck speeds
// / 1.5 rad/s (3), neck efforts on -1..1 (3), the chest's view of world up (3), the four support
// shares and the support total (5), effort power / 66 W, holding effort / 63 N m, limit load,
// capacity minimum and mean (5), and the valid flag (field 28, 1.0 on a live read). The frozen
// statistics: mean [28] and variance [28] (or spread [28]) and the weight, from the checkpoint's
// runtime_state.pt `vestibular_norm` (vestibular_fields.md section 3; the WI-34 fixtures carry
// `vestibular_norm/mean`, `/spread` and `/weight`).
//
// OPERATION (`VestibularNormaliser.batch` and `.transform`, Source/Python/mimo_bridge/
// vestibular_sense.py:435-465), in float64, stored float32:
//  1. x = the first 28 fields (every field but `valid`, `ENCODER_FIELDS`, :129-131), each clipped
//     to -4..4 (`INPUT_BOUND` 4.0, :172; :453). A non-finite field is missing (read_frames,
//     :398-400).
//  2. With weight <= 0 (no statistics yet) every value is 0 (:440-441). Otherwise
//     z = (x - mean) / spread, spread = sqrt(max(variance, 1e-6)) (`SPREAD_FLOOR` 1e-3 squared,
//     :150, :435-436); a missing or non-finite z is 0, the running mean (:443-444).
//  3. z clipped to -4..4 (`NORMALISED_BOUND` 4.0, :155; :447).
//  The read is real when all 29 fields are finite and `valid` > 0.5 (:398-405); a reading that
//  is not real is still encoded and marked not fresh ("stale", vestibular_fields.md section 2).
//  The statistics never move here: in an inference session nothing calls `commit`
//  (vestibular_fields.md section 3, "the Swift normaliser is three frozen arrays plus the two
//  bounds").
//
// OUTPUT. float32 [28], the vestibular encoder's input.

import Foundation
import UEMACore

/// The vestibular tokenizer: 28 fields centred and scaled by the checkpoint's frozen mean and
/// spread, then bounded (vestibular_sense.py:410-465).
public struct VestibularNormaliser: Tokenizer, Sendable {
    /// Fields per reading: 29 (`VESTIBULAR_DIM`, vestibular_adapter.py:40-53).
    public static let fieldCount = 29

    /// Encoder inputs: 28 (`ENCODER_DIM`, vestibular_sense.py:130).
    public static let encoderFieldCount = 28

    /// The valid-flag field: 28.
    public static let validField = 28

    /// The raw bound, in each field's own units: 4.0 (`INPUT_BOUND`, vestibular_sense.py:172).
    public static let inputBound: Double = 4

    /// The bound on a centred value, in spreads: 4.0 (`NORMALISED_BOUND`, vestibular_sense.py:155).
    public static let normalisedBound: Double = 4

    /// The smallest spread a field is divided by: 1e-3 (`SPREAD_FLOOR`, vestibular_sense.py:150).
    public static let spreadFloor: Double = 1e-3

    /// The sense served: vestibular.
    public var sense: SenseName { .vestibular }

    /// The running mean of each of the 28 fields, frozen.
    public let mean: [Double]

    /// The spread each field is divided by: sqrt(max(variance, 1e-6)).
    public let spread: [Double]

    /// The share of the 2,000-step horizon the statistics filled (0 = none yet; vestibular_sense.py:426).
    public let weight: Double

    /// Statistics from the checkpoint's mean and variance (runtime_state.pt `vestibular_norm`).
    /// Throws `TokenizerError.wrongStatisticsCount` for arrays of other than 28 numbers.
    public init(mean: [Double], variance: [Double], weight: Double) throws {
        try VestibularNormaliser.check("mean", mean)
        try VestibularNormaliser.check("variance", variance)
        self.mean = mean
        self.spread = variance.map { max($0, VestibularNormaliser.spreadFloor * VestibularNormaliser.spreadFloor).squareRoot() }
        self.weight = weight
    }

    /// Statistics from the mean and the already-floored spread (the WI-34 fixtures'
    /// `vestibular_norm/mean` and `/spread`). Throws for arrays of other than 28 numbers.
    public init(mean: [Double], spread: [Double], weight: Double) throws {
        try VestibularNormaliser.check("mean", mean)
        try VestibularNormaliser.check("spread", spread)
        self.mean = mean
        self.spread = spread.map { max($0, VestibularNormaliser.spreadFloor) }
        self.weight = weight
    }

    /// The normaliser before any real frame: weight 0, so every input is 0 (vestibular_sense.py:423-426).
    public static let cold = VestibularNormaliser(
        uncheckedMean: [Double](repeating: 0, count: encoderFieldCount),
        spread: [Double](repeating: spreadFloor, count: encoderFieldCount), weight: 0)

    private init(uncheckedMean mean: [Double], spread: [Double], weight: Double) {
        self.mean = mean
        self.spread = spread
        self.weight = weight
    }

    private static func check(_ name: String, _ values: [Double]) throws {
        guard values.count == encoderFieldCount else {
            throw TokenizerError.wrongStatisticsCount(name: name, expected: encoderFieldCount, found: values.count)
        }
    }

    /// INPUT: `fields` [29] as this file's header lists. OUTPUT: the 28 encoder inputs and
    /// whether the read is real (steps 1 to 3 of the header).
    public func normalise(_ fields: [Float]) throws -> (values: [Float], isRealRead: Bool) {
        guard fields.count == VestibularNormaliser.fieldCount else {
            throw TokenizerError.wrongShape(
                sense: .vestibular, array: "primary", expected: "[29]", found: [fields.count])
        }
        let allFinite = fields.allSatisfy { $0.isFinite }
        let isReal = allFinite && Double(fields[VestibularNormaliser.validField]) > 0.5
        var out = [Float](repeating: 0, count: VestibularNormaliser.encoderFieldCount)
        guard weight > 0 else {
            return (out, isReal)
        }
        let bound = VestibularNormaliser.normalisedBound
        for i in 0 ..< VestibularNormaliser.encoderFieldCount {
            let raw = Double(fields[i])
            guard raw.isFinite else { continue }
            let x = min(max(raw, -VestibularNormaliser.inputBound), VestibularNormaliser.inputBound)
            let z = (x - mean[i]) / spread[i]
            guard z.isFinite else { continue }
            out[i] = Float(min(max(z, -bound), bound))
        }
        return (out, isReal)
    }

    /// The encoder input [28] of one reading; fresh only when the reading is fresh and real.
    public func tokenize(_ reading: SenseReading) throws -> EncoderInput {
        try TokenizerChecks.requireSense(reading, .vestibular)
        guard reading.primary.rank == 1 else {
            throw TokenizerError.wrongShape(
                sense: .vestibular, array: "primary", expected: "[29]", found: reading.primary.shape)
        }
        let result = try normalise(reading.primary.values)
        return EncoderInput(
            sense: .vestibular,
            primary: try ShapedArray(shape: [VestibularNormaliser.encoderFieldCount], values: result.values),
            isFresh: reading.isFresh && result.isRealRead)
    }
}
