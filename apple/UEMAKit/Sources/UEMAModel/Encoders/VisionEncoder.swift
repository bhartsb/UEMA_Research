// VisionEncoder.swift: the vision sense's encoder of the released model in MLX Swift (work item
// WI-35 of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-9).
// Nothing in this file was compiled on the machine that wrote it (no Swift toolchain on Windows).
//
// INPUT: the vision encoder input of one model step (UEMATokenizer's VisionPatcher, WI-30): patches
//   [N, 768] (16 x 16 x 3 pixels divided by 255) and their angles [N, 4] (azimuth, elevation, eye
//   -1/+1, span); N = 1,160 on the released cameras (Tests/UEMAModelTests/Fixtures/model_pass,
//   input/vision/patches [1, 1160, 768]).
// OPERATION (lem/model/encoders/swin_patch.py, read November 4, 2026):
//   1. x = patch_embed(patches): Linear(768 -> 192, no bias) (swin_patch.py:205, 244; bias false
//      from encoder_defaults, lem/master_config.yaml:265).
//   2. x = x + angular_pos_enc(angles): Linear(4 -> 192, bias), GELU, Linear(192 -> 192, bias)
//      (swin_patch.py:33-43, 246-247; nn.Linear's own default bias=True).
//   3. No subsample: patch_subsample 4,096 (mimo_bridge/lem_model.py LITE_OVERRIDES) is above
//      1,160 (swin_patch.py:249).
//   4. 4 Swin blocks (args.json vision_swin_layers 4), width 192, 8 heads, window 7
//      (lem/master_config.yaml:73-75): pad the sequence with zero rows to a multiple of 7; odd
//      blocks (1 and 3) roll the padded sequence by -3 before cutting it into windows of 7 and roll
//      back by +3 after (swin_patch.py:85-94, 113-114, 221-222); in each window, x + attn(norm1(x))
//      then + ffn(norm2(.)) with ffn Linear(192, 768), GELU, Linear(768, 192), no biases
//      (swin_patch.py:100-109); the padding rows are cut off at the end of every block
//      (swin_patch.py:115-116).
//   5. Pool (swin_patch.py:121-151): q = LayerNorm(query [1, 1, 192]); q = q / |q| x 192^0.25;
//      k = x / |x| x 192^0.25 per patch; out = attn(q, k, x) with 4 heads (encoder_defaults
//      pool_n_heads 4, lem/master_config.yaml:270), no biases. proj is the identity
//      (d_internal 192 = d_output 192, swin_patch.py:228-229).
// OUTPUT: [1, 1, 192], the pooled vector before the registry's shared output LayerNorm (the fixture's
//   encoder_output/vision); backbone(...) gives the per-patch features [1, N, 192]
//   (encoder_features/vision).
// PARAMETERS: 147,456 + 38,016 + 4 x 443,136 + 148,032 = 2,106,048 (weight_manifest.json component
//   enc_vision; model_receipt.json component_params).
// FIRST BUILD ON A MAC: see TransformerCore.swift, steps 1 to 4; ModelPassTests checks this
//   encoder's parameter count and keys without the weights, and its output against
//   encoder_output/vision with them.

import Foundation
import MLX
import UEMACore

/// The vision encoder (SwinPatchEncoder of swin_patch.py:184-281 with the released sizes).
public struct VisionEncoder: SenseEncoding {
    /// The released sizes.
    public enum Released {
        /// Numbers per patch: 16 x 16 x 3 = 768.
        public static let patchWidth = 768
        /// Angle numbers per patch: 4.
        public static let angleCount = 4
        /// Internal and output width: 192.
        public static let width = 192
        /// Swin blocks: 4.
        public static let blocks = 4
        /// Heads per Swin block: 8.
        public static let heads = 8
        /// Patches per window: 7.
        public static let windowSize = 7
        /// Heads of the pooling attention: 4.
        public static let poolHeads = 4
        /// Parameters: 2,106,048.
        public static let parameterCount = 2_106_048
    }

    public let sense: SenseName = .vision
    let patchEmbed: LinearLayer
    let angleIn: LinearLayer
    let angleOut: LinearLayer
    let blocks: [SwinBlock]
    let pool: UnitLengthPool

    /// The encoder stored under `source` (the released file seen through
    /// `scoped("encoder_registry.encoders.vision.")`).
    public init(source: any MLXTensorSource) throws {
        let width = Released.width
        patchEmbed = try LinearLayer(source: source, prefix: "patch_embed.", inputs: Released.patchWidth,
                                     outputs: width, hasBias: false)
        angleIn = try LinearLayer(source: source, prefix: "angular_pos_enc.mlp.0.", inputs: Released.angleCount,
                                  outputs: width, hasBias: true)
        angleOut = try LinearLayer(source: source, prefix: "angular_pos_enc.mlp.2.", inputs: width,
                                   outputs: width, hasBias: true)
        blocks = try (0 ..< Released.blocks).map { index in
            SwinBlock(block: try PreNormAttentionBlock(source: source, prefix: "blocks.\(index).",
                                                       attentionNormName: "norm1", feedForwardNormName: "norm2",
                                                       width: width, heads: Released.heads, expansion: 4,
                                                       hasBias: false),
                      windowSize: Released.windowSize, shifted: index % 2 == 1)
        }
        pool = try UnitLengthPool(source: source, prefix: "pool.", width: width, heads: Released.poolHeads)
    }

    public var parameterCount: Int {
        patchEmbed.parameterCount + angleIn.parameterCount + angleOut.parameterCount
            + blocks.reduce(0) { $0 + $1.block.parameterCount } + pool.parameterCount
    }

    /// Steps 1 to 4: patches [B, N, 768] and angles [B, N, 4] into features [B, N, 192].
    public func backbone(patches: MLXArray, angles: MLXArray) -> MLXArray {
        var x = patchEmbed(patches) + angleOut(NeuralOps.gelu(angleIn(angles)))
        for block in blocks {
            x = block(x)
        }
        return x
    }

    /// Step 5: features [B, N, 192] into the pooled vector [B, 1, 192].
    public func head(_ features: MLXArray) -> MLXArray {
        pool(features)
    }

    /// The pooled vector [B, 1, 192] of patches [B, N, 768] with angles [B, N, 4].
    public func callAsFunction(patches: MLXArray, angles: MLXArray) -> MLXArray {
        head(backbone(patches: patches, angles: angles))
    }

    public func encode(_ input: EncoderInput) throws -> MLXArray {
        guard input.sense == .vision else {
            throw ModelPortError.missingInput(.vision)
        }
        let patches = try NeuralOps.batched(input.primary, expected: [-1, Released.patchWidth], what: "vision patches")
        guard let angleArray = input.auxiliary else {
            throw ModelPortError.inputShape(what: "vision angles", expected: [input.primary.shape[0], Released.angleCount],
                                            found: [])
        }
        let angles = try NeuralOps.batched(angleArray, expected: [input.primary.shape[0], Released.angleCount],
                                           what: "vision angles")
        return self(patches: patches, angles: angles)
    }
}

/// One Swin block over the flat patch sequence (swin_patch.py:46-118): zero padding to a multiple
/// of the window, the half-window roll on shifted blocks, the pre-norm attention block inside each
/// window, the roll back and the cut of the padding.
struct SwinBlock {
    let block: PreNormAttentionBlock
    let windowSize: Int
    let shifted: Bool

    /// The block on x [B, N, D].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.shape[0]
        let count = x.shape[1]
        let width = x.shape[2]
        let padding = (windowSize - count % windowSize) % windowSize
        var padded = x
        if padding > 0 {
            padded = concatenated([x, zeros([batch, padding, width], type: Float.self)], axis: 1)
        }
        let paddedCount = count + padding
        let shift = shifted ? windowSize / 2 : 0
        if shift > 0 {
            padded = roll(padded, shift: -shift, axis: 1)
        }
        let windows = padded.reshaped([batch * (paddedCount / windowSize), windowSize, width])
        var out = block(windows).reshaped([batch, paddedCount, width])
        if shift > 0 {
            out = roll(out, shift: shift, axis: 1)
        }
        if padding > 0 {
            out = out[0..., 0 ..< count]
        }
        return out
    }
}

/// The vision pool (LearnedPoolQuery, swin_patch.py:121-151): one learned query, LayerNormed, both
/// query and keys scaled to length width^0.25 before the attention's projections, the values the
/// features themselves.
struct UnitLengthPool {
    let query: MLXArray
    let norm: LayerNormLayer
    let attention: MultiheadAttentionLayer
    let lengthScale: Float

    init(source: any MLXTensorSource, prefix: String, width: Int, heads: Int) throws {
        query = try source.tensor(prefix + "query", shape: [1, 1, width])
        norm = try LayerNormLayer(source: source, prefix: prefix + "norm.", width: width)
        attention = try MultiheadAttentionLayer(source: source, prefix: prefix + "attn.", width: width,
                                                heads: heads, hasBias: false)
        // width^0.25 as two square roots: 192^0.25 = 3.7224 (swin_patch.py:148-149, q.shape[-1] ** 0.25)
        lengthScale = Float(width).squareRoot().squareRoot()
    }

    var parameterCount: Int { query.size + norm.parameterCount + attention.parameterCount }

    /// features [B, N, D] into [B, 1, D].
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        let batch = features.shape[0]
        let width = features.shape[2]
        let q = NeuralOps.unitLength(norm(query)) * lengthScale
        let keys = NeuralOps.unitLength(features) * lengthScale
        return attention(query: broadcast(q, to: [batch, 1, width]), key: keys, value: features)
    }
}
