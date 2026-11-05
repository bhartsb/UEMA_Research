// TactileEncoder.swift: the tactile sense's encoder of the released model in MLX Swift, and the
// cross-patch attention encoder it shares with the motor sense (work item WI-35 of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-9). Nothing in
// this file was compiled on the machine that wrote it (no Swift toolchain on Windows).
//
// INPUT: the tactile encoder input of one model step (UEMATokenizer's TactileTokenizer, WI-30;
//   apple/UEMAKit/Specs/touch_pooler.md): 32 contact points x 19 numbers [32, 19] (the fixture's
//   input/tactile [1, 32, 19]).
// OPERATION (lem/model/encoders/cross_patch.py, CrossPatchAttentionEncoder, read November 4, 2026):
//   1. x = x x input_scale with input_scale 1.0 (lem/master_config.yaml:100), so no change
//      (cross_patch.py:97-98).
//   2. patch_mlp: Linear(19 -> 192, no bias), GELU, Linear(192 -> 192, no bias)
//      (cross_patch.py:50-54; per_patch_hidden 192 = args.json encoder_width 192 through
//      mimo_bridge/train_mimo.py size_overrides; bias false from encoder_defaults,
//      lem/master_config.yaml:265).
//   3. No per-slot position vector for tactile (patch_pos_embedding False in LITE_OVERRIDES,
//      mimo_bridge/lem_model.py; the manifest has no encoder_registry.encoders.tactile.patch_pos).
//      The motor encoder adds patch_pos [1, 26, 192] to the first min(N, 26) rows
//      (cross_patch.py:100-102).
//   4. 2 layers (args.json encoder_layers 2), each x + attn(norm(x)), then + ffn(ffn_norm(.)),
//      4 heads (lem/master_config.yaml:104), ffn Linear(192, 768), GELU, Linear(768, 192), no
//      biases (cross_patch.py:60-76, 104-112). CappedMultiheadAttention without a bound is
//      nn.MultiheadAttention (attention_cap.py:29; args.json attention_cap off).
//   5. Pool: q = pool_norm(pool_query [1, 1, 192]); out = pool_attn(q, x, x) with 4 heads
//      (pool_n_heads 4, lem/master_config.yaml:270); proj is the identity (cross_patch.py:116-124).
//   6. The liveness repair's all-element path (args.json liveness_repair on; lem/model/liveness.py
//      226-227 POOL_PATHS ("motor", "tactile") with rank 19, 454-475 AllElementPath, 729-735 the
//      forward hook on pool_attn that returns its output + path(keys)): each element of the pool's
//      keys x [B, N, 192] through LayerNorm(192) and down Linear(192 -> 19, no bias), the N x 19
//      numbers concatenated and mapped by up Linear(N x 19 -> 192, bias) to [B, 1, 192], added to the
//      pool's output. Its weights are "liveness_repair.paths.<sense>." in model.safetensors (component
//      liveness_repair, outside the manifest's inference set; N = 32 for tactile, up.weight
//      [192, 608]; N = 26 for motor, [192, 494]). MEASURED: without it the pooled vector differed from
//      the fixture's encoder_output by up to 4,782 (tactile) and 9,742 (motor) on model step 492,800,
//      and the tokens by up to 0.13 and 1.44; with it a float64 numpy copy of this arithmetic matched
//      the tokens within 8e-7 on the three WI-34 fixtures.
// OUTPUT: [1, 1, 192] before the shared output LayerNorm (the fixture's encoder_output/tactile);
//   backbone(...) gives the per-point features [1, 32, 192] (encoder_features/tactile).
// PARAMETERS: tactile 40,512 + 2 x 443,136 + 148,032 = 1,074,816 (manifest component enc_tactile);
//   motor adds patch_pos 4,992: 1,079,808 (enc_motor).
// FIRST BUILD ON A MAC: see TransformerCore.swift, steps 1 to 4.

import Foundation
import MLX
import UEMACore

/// The cross-patch attention encoder (cross_patch.py:24-127) at the released sizes, shared by the
/// tactile and motor senses.
public struct CrossPatchAttentionEncoder {
    /// Internal and output width: 192.
    public static let width = 192
    /// Attention layers: 2 (args.json encoder_layers 2).
    public static let layerCount = 2
    /// Heads per attention layer: 4.
    public static let heads = 4
    /// Heads of the pooling attention: 4.
    public static let poolHeads = 4

    /// Numbers per row of the input: 19.
    public let inputWidth: Int
    let patchIn: LinearLayer
    let patchOut: LinearLayer
    /// The per-slot position vectors [1, slots, 192]; nil for tactile.
    let patchPosition: MLXArray?
    let layers: [PreNormAttentionBlock]
    let poolQuery: MLXArray
    let poolNorm: LayerNormLayer
    let poolAttention: MultiheadAttentionLayer
    /// The liveness repair's all-element path beside the pool; nil adds nothing.
    let poolPath: AllElementPath?

    /// The encoder stored under `source`; `positionSlots` > 0 reads patch_pos [1, positionSlots, 192];
    /// `poolPath` is added to the pool's output.
    public init(source: any MLXTensorSource, inputWidth: Int, positionSlots: Int,
                poolPath: AllElementPath? = nil) throws {
        let width = CrossPatchAttentionEncoder.width
        self.inputWidth = inputWidth
        patchIn = try LinearLayer(source: source, prefix: "patch_mlp.0.", inputs: inputWidth, outputs: width,
                                  hasBias: false)
        patchOut = try LinearLayer(source: source, prefix: "patch_mlp.2.", inputs: width, outputs: width,
                                   hasBias: false)
        if positionSlots > 0 {
            patchPosition = try source.tensor("patch_pos", shape: [1, positionSlots, width])
        } else {
            patchPosition = nil
        }
        layers = try (0 ..< CrossPatchAttentionEncoder.layerCount).map { index in
            try PreNormAttentionBlock(source: source, prefix: "layers.\(index).", attentionNormName: "norm",
                                      feedForwardNormName: "ffn_norm", width: width,
                                      heads: CrossPatchAttentionEncoder.heads, expansion: 4, hasBias: false)
        }
        poolQuery = try source.tensor("pool_query", shape: [1, 1, width])
        poolNorm = try LayerNormLayer(source: source, prefix: "pool_norm.", width: width)
        poolAttention = try MultiheadAttentionLayer(source: source, prefix: "pool_attn.", width: width,
                                                    heads: CrossPatchAttentionEncoder.poolHeads, hasBias: false)
        self.poolPath = poolPath
    }

    /// The parameter count.
    public var parameterCount: Int {
        patchIn.parameterCount + patchOut.parameterCount + (patchPosition?.size ?? 0)
            + layers.reduce(0) { $0 + $1.parameterCount } + poolQuery.size + poolNorm.parameterCount
            + poolAttention.parameterCount
    }

    /// Steps 1 to 4: rows [B, N, inputWidth] into features [B, N, 192].
    public func backbone(_ rows: MLXArray) -> MLXArray {
        var x = patchOut(NeuralOps.gelu(patchIn(rows)))
        if let patchPosition {
            let count = x.shape[1]
            let slots = patchPosition.shape[1]
            let shared = min(count, slots)
            let position = shared == slots ? patchPosition : patchPosition[0..., 0 ..< shared]
            if shared == count {
                x = x + position
            } else {
                x = concatenated([x[0..., 0 ..< shared] + position, x[0..., shared...]], axis: 1)
            }
        }
        for layer in layers {
            x = layer(x)
        }
        return x
    }

    /// Step 5: features [B, N, 192] into the pooled vector [B, 1, 192].
    public func head(_ features: MLXArray) -> MLXArray {
        let batch = features.shape[0]
        let query = broadcast(poolNorm(poolQuery), to: [batch, 1, CrossPatchAttentionEncoder.width])
        let pooled = poolAttention(query: query, key: features, value: features)
        guard let poolPath else {
            return pooled
        }
        return pooled + poolPath(features)
    }

    /// The pooled vector [B, 1, 192] of rows [B, N, inputWidth].
    public func callAsFunction(_ rows: MLXArray) -> MLXArray {
        head(backbone(rows))
    }
}

/// The tactile encoder: the cross-patch attention encoder over 32 contact points x 19 numbers.
public struct TactileEncoder: SenseEncoding {
    /// Contact points per model step: 32.
    public static let points = 32
    /// Numbers per point: 19.
    public static let numbersPerPoint = 19
    /// Parameters: 1,074,816.
    public static let parameterCount = 1_074_816

    public let sense: SenseName = .tactile
    /// The shared encoder body.
    public let core: CrossPatchAttentionEncoder

    /// The encoder stored under `source` (the released file seen through
    /// `scoped("encoder_registry.encoders.tactile.")`), with the liveness repair's pool path when given
    /// (the released run's; AllElementPath.released(sense: .tactile, ...)).
    public init(source: any MLXTensorSource, poolPath: AllElementPath? = nil) throws {
        core = try CrossPatchAttentionEncoder(source: source, inputWidth: TactileEncoder.numbersPerPoint,
                                              positionSlots: 0, poolPath: poolPath)
    }

    public var parameterCount: Int { core.parameterCount }

    /// The pooled vector [B, 1, 192] of points [B, 32, 19].
    public func callAsFunction(_ points: MLXArray) -> MLXArray {
        core(points)
    }

    public func encode(_ input: EncoderInput) throws -> MLXArray {
        guard input.sense == .tactile else {
            throw ModelPortError.missingInput(.tactile)
        }
        let points = try NeuralOps.batched(input.primary, expected: [TactileEncoder.points, TactileEncoder.numbersPerPoint],
                                           what: "tactile points")
        return core(points)
    }
}

/// The liveness repair's all-element path beside an attention pool (AllElementPath of
/// lem/model/liveness.py:454-475): LayerNorm each of the N elements, project it to `rank` numbers with
/// one shared matrix, and map the N x rank concatenation (each element through its own columns) to
/// the pool's width.
public struct AllElementPath {
    /// Numbers kept per element: 19 for both pools (PATH_RANK, liveness.py:227).
    public static let releasedRank = 19

    /// Elements the path reads: 26 for motor, 32 for tactile.
    public let elements: Int
    /// Numbers kept per element.
    public let rank: Int
    let norm: LayerNormLayer
    let down: LinearLayer
    let up: LinearLayer

    /// The path stored under `source` (the released file seen through
    /// `scoped("liveness_repair.paths.<sense>.")`).
    public init(source: any MLXTensorSource, elements: Int, rank: Int = AllElementPath.releasedRank,
                width: Int = CrossPatchAttentionEncoder.width) throws {
        self.elements = elements
        self.rank = rank
        norm = try LayerNormLayer(source: source, prefix: "norm.", width: width)
        down = try LinearLayer(source: source, prefix: "down.", inputs: width, outputs: rank, hasBias: false)
        up = try LinearLayer(source: source, prefix: "up.", inputs: elements * rank, outputs: width, hasBias: true)
    }

    /// The released path of `sense` (tactile with 32 elements, motor with 26) from the weights; nil
    /// when the weights hold none (a build without the repair).
    public static func released(sense: SenseName, elements: Int,
                                from weights: any MLXTensorSource) throws -> AllElementPath? {
        let prefix = "liveness_repair.paths.\(sense.rawValue)."
        guard weights.hasTensor(prefix + "up.weight") else {
            return nil
        }
        return try AllElementPath(source: weights.scoped(prefix), elements: elements)
    }

    /// The parameter count: 384 + 19 x 192 + (N x 19 + 1) x 192.
    public var parameterCount: Int { norm.parameterCount + down.parameterCount + up.parameterCount }

    /// keys [B, N, width] into the addition [B, 1, width].
    public func callAsFunction(_ keys: MLXArray) -> MLXArray {
        let batch = keys.shape[0]
        let reduced = down(norm(keys))
        return up(reduced.reshaped([batch, 1, elements * rank]))
    }
}
