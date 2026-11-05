// VestibularEncoder.swift: the vestibular sense's encoder of the released model in MLX Swift (work
// item WI-35 of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-9).
// Nothing in this file was compiled on the machine that wrote it (no Swift toolchain on Windows).
//
// INPUT: 28 head-motion, neck, support and effort fields, centred and scaled by the normaliser's
//   frozen mean and spread [28] (UEMATokenizer's VestibularNormaliser, WI-30;
//   apple/UEMAKit/Specs/vestibular_fields.md; the fixture's input/vestibular [1, 28], with the
//   statistics in vestibular_norm/mean and vestibular_norm/spread).
// OPERATION: the multi-layer perceptron encoder of ProprioceptionEncoder.swift with 28 inputs and
//   Linear biases (mimo_bridge/vestibular_sense.py:276-297: "type": "mlp_project", the hidden
//   dimensions of proprioception, "bias": True). The released run's --vestibular-encoder-repair is
//   rescale_restart (args.json): the division of the last layer happened at an earlier load and is
//   inside the stored weights; the fp32 forward of vestibular_sense.py:1004-1023 applies only to the
//   *_fp32 modes and changes no number on a float32 cpu pass. --vestibular-input-path is off
//   (args.json), so no linear path from the 28 inputs is added.
// OUTPUT: [B, 1, 192] before the shared output LayerNorm (the fixture's encoder_output/vestibular).
// PARAMETERS: 5,568 + 384 + 2 x (37,056 + 384) + 37,056 = 117,888 (manifest component enc_vestibular).
// FIRST BUILD ON A MAC: see TransformerCore.swift, steps 1 to 4.

import Foundation
import MLX
import UEMACore

/// The vestibular encoder: the multi-layer perceptron encoder over 28 fields, with Linear biases.
public struct VestibularEncoder: SenseEncoding {
    /// Fields per model step: 28.
    public static let inputWidth = 28
    /// Parameters: 117,888.
    public static let parameterCount = 117_888

    public let sense: SenseName = .vestibular
    /// The shared encoder body.
    public let core: MLPProjectEncoder

    /// The encoder stored under `source` (the released file seen through
    /// `scoped("encoder_registry.encoders.vestibular.")`).
    public init(source: any MLXTensorSource) throws {
        core = try MLPProjectEncoder(source: source, inputWidth: VestibularEncoder.inputWidth, hasBias: true)
    }

    public var parameterCount: Int { core.parameterCount }

    /// The pooled vector [B, 1, 192] of fields [B, 28].
    public func callAsFunction(_ fields: MLXArray) -> MLXArray {
        core(fields)
    }

    public func encode(_ input: EncoderInput) throws -> MLXArray {
        guard input.sense == .vestibular else {
            throw ModelPortError.missingInput(.vestibular)
        }
        let fields = try NeuralOps.batched(input.primary, expected: [VestibularEncoder.inputWidth],
                                           what: "vestibular fields")
        return core(fields)
    }
}
