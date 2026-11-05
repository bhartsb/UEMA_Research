// ProprioceptionEncoder.swift: the proprioception sense's encoder of the released model in MLX
// Swift, and the multi-layer perceptron encoder it shares with the vestibular sense (work item WI-35
// of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-9). Nothing in
// this file was compiled on the machine that wrote it (no Swift toolchain on Windows).
//
// INPUT: 26 handle rows x 12 numbers flattened to [312] (UEMATokenizer's ProprioceptionTokenizer,
//   WI-30; apple/UEMAKit/Specs/proprio_rows.md; the fixture's input/proprioception [1, 312]).
// OPERATION (lem/model/encoders/mlp_project.py, MLPProjectEncoder, read November 4, 2026): for each
//   of the 3 hidden layers (hidden_dims [192] x (1 + encoder_layers 2), size overrides of
//   mimo_bridge/train_mimo.py; Sequential indices 4i, 4i + 1, 4i + 2, 4i + 3): Linear, LayerNorm,
//   GELU (encoder_defaults activation gelu, lem/master_config.yaml:267), Dropout 0.0; then the
//   final Linear(192 -> 192) at index 12 (mlp_project.py:326-340). Proprioception has no Linear
//   biases (encoder_defaults bias false, master_config.yaml:265); the vestibular encoder has them
//   (mimo_bridge/vestibular_sense.py:296-297, "bias": True). The LayerNorms always carry weight and
//   bias. The output gains a sequence extent of 1 (mlp_project.py:349-350).
//   The released run's --proprio-heading is off (args.json proprio_heading off), so no hook turns
//   the rows before this encoder (mimo_bridge/lem_model.py:1673-1688).
// OUTPUT: [B, 1, 192] before the shared output LayerNorm (the fixture's encoder_output/proprioception).
// PARAMETERS: 59,904 + 384 + 2 x (36,864 + 384) + 36,864 = 171,648 (manifest component
//   enc_proprioception).
// FIRST BUILD ON A MAC: see TransformerCore.swift, steps 1 to 4.

import Foundation
import MLX
import UEMACore

/// The multi-layer perceptron encoder (mlp_project.py:309-350) at the released sizes, shared by the
/// proprioception and vestibular senses.
public struct MLPProjectEncoder {
    /// Hidden layers: 3.
    public static let hiddenLayers = 3
    /// Width of every hidden layer and of the output: 192.
    public static let width = 192

    /// Numbers of the flat input.
    public let inputWidth: Int
    let hidden: [(linear: LinearLayer, norm: LayerNormLayer)]
    let output: LinearLayer

    /// The encoder stored under `source` with `inputWidth` inputs; `hasBias` for the Linear layers.
    public init(source: any MLXTensorSource, inputWidth: Int, hasBias: Bool) throws {
        let width = MLPProjectEncoder.width
        self.inputWidth = inputWidth
        var layers: [(linear: LinearLayer, norm: LayerNormLayer)] = []
        var inputs = inputWidth
        for layer in 0 ..< MLPProjectEncoder.hiddenLayers {
            let linear = try LinearLayer(source: source, prefix: "mlp.\(4 * layer).", inputs: inputs,
                                         outputs: width, hasBias: hasBias)
            let norm = try LayerNormLayer(source: source, prefix: "mlp.\(4 * layer + 1).", width: width)
            layers.append((linear: linear, norm: norm))
            inputs = width
        }
        hidden = layers
        output = try LinearLayer(source: source, prefix: "mlp.\(4 * MLPProjectEncoder.hiddenLayers).",
                                 inputs: width, outputs: width, hasBias: hasBias)
    }

    /// The parameter count.
    public var parameterCount: Int {
        hidden.reduce(0) { $0 + $1.linear.parameterCount + $1.norm.parameterCount } + output.parameterCount
    }

    /// The pooled vector [B, 1, 192] of flat inputs [B, inputWidth].
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in hidden {
            h = NeuralOps.gelu(layer.norm(layer.linear(h)))
        }
        return output(h).expandedDimensions(axis: 1)
    }
}

/// The proprioception encoder: the multi-layer perceptron encoder over 312 numbers, no Linear biases.
public struct ProprioceptionEncoder: SenseEncoding {
    /// Numbers of the flat input: 26 rows x 12 = 312.
    public static let inputWidth = 312
    /// Parameters: 171,648.
    public static let parameterCount = 171_648

    public let sense: SenseName = .proprioception
    /// The shared encoder body.
    public let core: MLPProjectEncoder

    /// The encoder stored under `source` (the released file seen through
    /// `scoped("encoder_registry.encoders.proprioception.")`).
    public init(source: any MLXTensorSource) throws {
        core = try MLPProjectEncoder(source: source, inputWidth: ProprioceptionEncoder.inputWidth, hasBias: false)
    }

    public var parameterCount: Int { core.parameterCount }

    /// The pooled vector [B, 1, 192] of rows [B, 312].
    public func callAsFunction(_ rows: MLXArray) -> MLXArray {
        core(rows)
    }

    public func encode(_ input: EncoderInput) throws -> MLXArray {
        guard input.sense == .proprioception else {
            throw ModelPortError.missingInput(.proprioception)
        }
        let rows = try NeuralOps.batched(input.primary, expected: [ProprioceptionEncoder.inputWidth],
                                         what: "proprioception rows")
        return core(rows)
    }
}
