// PredictionHeads.swift: the prediction heads of the released model in MLX Swift, the heads' all-slot
// path the released run adds to them, and the liveness repair's revived units inside both (work item
// WI-35 of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-10).
// Nothing in this file was compiled on the machine that wrote it (no Swift toolchain on Windows).
//
// INPUT: the transformer's last model step, 6 tokens [B, 6, 192] (TransformerOutput.lastStep,
//   memory bias included), and this model step's raw tokens per sense [B, 192] (after the shared
//   output LayerNorm, before any top-down blend: lem_model.py "current={m: tokens_raw[m]...}").
// OPERATION (lem/model/prediction_heads.py, read November 4, 2026):
//   one head per sense except motor, in the sequence vision, tactile, audio, proprioception,
//   vestibular (model_receipt.json prediction_head_modalities; the fixture's metadata
//   prediction_senses). Each head (prediction_heads.py:19-62): q = LayerNorm(query [1, 1, 192]);
//   a = attn(q, lastStep, lastStep) with 4 heads and biases (lem/master_config.yaml:295;
//   nn.MultiheadAttention default bias=True); mlp = Linear(192 -> 192), GELU, Linear(192 -> 192)
//   (n_mlp_layers 2, hidden_expansion 1 from LITE_OVERRIDES, mimo_bridge/lem_model.py). The vision
//   head's first Linear reads a / input_scale (lem/model/input_conditioning.py:14-22
//   InputScaledLinear, buffer prediction_heads.heads.vision.mlp.0.input_scale, a scalar; receipt
//   visual_input_conditioning scale 261.7218). residual_prediction on (args.json): the head's
//   output plus the sense's raw token (prediction_heads.py:197, 240-242). Then the all-slot path's
//   output is added (prediction_heads.py:243-244, 248-255).
//   REVIVED UNITS (args.json liveness_repair on; lem/model/liveness.py:152-159 REPAIR_LAYERS and
//   290-361 RevivedRows; component liveness_repair, outside the manifest's inference set): the first
//   Linear of the vision, tactile and vestibular heads has all 192 of its 192 units revived (masks in
//   model.safetensors), so its output is lin(LayerNorm(a - centre)) of the repair's own weights, a
//   being the head's attention output as the Linear received it (before input_scale: the hook reads
//   the module's input). RevivedRowsLayer (TransformerCore.swift) applies it.
//   ALL-SLOT PATH (args.json heads_all_slot_path on; lem/model/heads_all_slot_path.py, read the same
//   day; component heads_all_slot_path, 403,200 parameters, outside the inference set; the fixture's
//   module_calls record heads_all_slot_path 1 call on the live pass): x = lastStep - c, c = the
//   persistent memory's constant [6, 192] = persistent_memory.readout.bias reshaped
//   (mimo_bridge/lem_model.py _asp_constant -> lem/model/motor_input_fold.py:95-104
//   effective_constant; persistent_memory_constant_return 0 in args.json, so no return factor
//   scales it); xc = x - centre_mean [6, 192] (a buffer in model.safetensors); per head
//   u = down(LayerNorm(xc)) [B, 6, 32] (down Linear(192 -> 32, no bias)), flattened to [B, 192];
//   h = net.0(u) with the revived units of net.0 replaced (139, 17, 87, 71 and 172 of 192 for
//   vision, tactile, audio, proprioception, vestibular); out = net.2(GELU(h)) / sqrt(192)
//   (heads_all_slot_path.py:135-144, 151-159, 227-235, 258-274, 284-297).
//   THE CENTRES of the eight revived layers are not in model.safetensors (persistent=False,
//   liveness.py:309); Resources/prediction_head_centres.json holds them, exported from
//   runtime_state.pt by Tests/UEMAModelTests/Fixtures/prediction_heads/export_prediction_head_centres.py.
// MEASURED BEFORE COMPILING (November 4, 2026): a float64 numpy copy of this file's arithmetic and
//   of the encoders and transformer (Tests/UEMAModelTests/Fixtures/prediction_heads/
//   check_model_pass_numpy.py, run on the three WI-34
//   fixtures with the released weights and these centres) reproduced every predicted token within
//   1.4e-6 and every raw token within 1.2e-6 (largest absolute difference per number; a real token
//   has a root mean square of 1.0). Without the revived units the vision prediction differed by up
//   to 0.28 and without the all-slot path by up to 1.37 on model step 492,800.
// OUTPUT: the predicted next token per sense [B, 192] for the five senses with a head (the fixture's
//   predicted/<sense>).
// PARAMETERS: prediction heads 5 x 222,912 = 1,114,560 (manifest component prediction_heads), plus
//   the one buffer input_scale; all-slot path 5 x 80,640 = 403,200; revived units 37,440 per layer of
//   192 inputs (norm 384, lin 37,056).
// FIRST BUILD ON A MAC: see TransformerCore.swift, steps 1 to 4.

import Foundation
import MLX
import UEMACore

// MARK: - The exported centres

/// The running centres of the revived layers the prediction path reads, from
/// Resources/prediction_head_centres.json (schema uema_prediction_head_centres_v1).
public struct PredictionHeadCentres: Decodable, Equatable, Sendable {
    /// The schema this reader reads.
    public static let expectedSchema = "uema_prediction_head_centres_v1"
    /// The resource's name in the module bundle.
    public static let resourceName = "prediction_head_centres"

    /// One layer's centre.
    public struct Layer: Decodable, Equatable, Sendable {
        /// The running mean's accumulated weight (0.99999... after a full horizon).
        public let centreWeight: Double
        /// The centre, one number per input of the layer.
        public let centreMean: [Float]

        private enum CodingKeys: String, CodingKey {
            case centreWeight = "centre_weight"
            case centreMean = "centre_mean"
        }
    }

    /// The schema name.
    public let schema: String
    /// The centres by the repair's layer key ("prediction_heads__heads__vision__mlp__0").
    public let layers: [String: Layer]

    /// Decodes `data`; another schema throws.
    public static func decode(from data: Data) throws -> PredictionHeadCentres {
        let centres = try JSONDecoder().decode(PredictionHeadCentres.self, from: data)
        guard centres.schema == expectedSchema else {
            throw ModelPortError.missingTensor(key: "\(resourceName).json schema \(centres.schema), expected \(expectedSchema)")
        }
        return centres
    }

    /// The bundled resource.
    public static func bundled(in bundle: Bundle = UEMAModelModule.resourceBundle) throws -> PredictionHeadCentres {
        guard let url = bundle.url(forResource: resourceName, withExtension: "json", subdirectory: "Resources") else {
            throw ModelPortError.missingTensor(key: "Resources/\(resourceName).json")
        }
        return try decode(from: Data(contentsOf: url))
    }

    /// The revived units of `layer` read from `weights` under "liveness_repair.rows.<layer>." with this
    /// file's centre; nil when the weights hold no mask for that layer (a build without the repair).
    public func revivedRows(layer: String, from weights: any MLXTensorSource, inputs: Int,
                            outputs: Int) throws -> RevivedRowsLayer? {
        let prefix = "liveness_repair.rows.\(layer)."
        guard weights.hasTensor(prefix + "mask") else {
            return nil
        }
        guard let entry = layers[layer] else {
            throw ModelPortError.missingTensor(key: "\(PredictionHeadCentres.resourceName).json layer \(layer)")
        }
        return try RevivedRowsLayer(source: weights, prefix: prefix, inputs: inputs, outputs: outputs,
                                    centre: entry.centreMean)
    }
}

// MARK: - One head

/// One prediction head (PredictionHead of prediction_heads.py:19-62).
public struct PredictionHead {
    /// Heads of the head's attention: 4.
    public static let attentionHeads = 4

    let query: MLXArray
    let norm: LayerNormLayer
    let attention: MultiheadAttentionLayer
    let hidden: LinearLayer
    let output: LinearLayer
    /// The revived units of the first Linear; nil when none.
    let revived: RevivedRowsLayer?
    /// The divisor of the first Linear's input: input_scale for vision, 1 for the others.
    public let inputScale: Float

    /// The head stored under `source` (the released file seen through
    /// `scoped("prediction_heads.heads.<sense>.")`); the scalar buffer "mlp.0.input_scale" is read
    /// when present; `revived` replaces units of the first Linear.
    public init(source: any MLXTensorSource, revived: RevivedRowsLayer? = nil,
                width: Int = TransformerCore.Released.width) throws {
        query = try source.tensor("query", shape: [1, 1, width])
        norm = try LayerNormLayer(source: source, prefix: "norm.", width: width)
        attention = try MultiheadAttentionLayer(source: source, prefix: "attn.", width: width,
                                                heads: PredictionHead.attentionHeads, hasBias: true)
        hidden = try LinearLayer(source: source, prefix: "mlp.0.", inputs: width, outputs: width, hasBias: true)
        output = try LinearLayer(source: source, prefix: "mlp.2.", inputs: width, outputs: width, hasBias: true)
        if source.hasTensor("mlp.0.input_scale") {
            inputScale = try source.tensor("mlp.0.input_scale", shape: []).item(Float.self)
        } else {
            inputScale = 1
        }
        self.revived = revived
    }

    /// The parameter count: 222,912 at width 192 (input_scale is a buffer; the revived units belong
    /// to the component liveness_repair and are not counted).
    public var parameterCount: Int {
        query.size + norm.parameterCount + attention.parameterCount + hidden.parameterCount + output.parameterCount
    }

    /// The head's own output [B, 192] for lastStep [B, 6, 192].
    public func callAsFunction(_ lastStep: MLXArray) -> MLXArray {
        let batch = lastStep.shape[0]
        let width = lastStep.shape[2]
        let q = broadcast(norm(query), to: [batch, 1, width])
        let attended = attention(query: q, key: lastStep, value: lastStep).reshaped([batch, width])
        // x / input_scale written as x times its reciprocal (an MLXArray times a ScalarOrArray)
        var pre = hidden(attended * (Float(1) / inputScale))
        if let revived {
            pre = revived.apply(input: attended, preActivation: pre)
        }
        return output(NeuralOps.gelu(pre))
    }
}

// MARK: - The all-slot path

/// Something added to every head's prediction after the residual: the released run's all-slot path.
public protocol PredictionAdding {
    /// The addition per sense [B, 192] for lastStep [B, 6, 192]; a sense left out adds nothing.
    func additions(lastStep: MLXArray) -> [SenseName: MLXArray]
}

/// The heads' all-slot path (HeadsAllSlotPath of heads_all_slot_path.py, inference arithmetic).
public struct HeadsAllSlotPath: PredictionAdding {
    /// Numbers kept per slot: 32 (SLOT_RANK, heads_all_slot_path.py:124).
    public static let slotRank = 32
    /// Parameters at width 192 with five heads: 403,200.
    public static let parameterCount = 403_200

    struct HeadPath {
        let norm: LayerNormLayer
        let down: LinearLayer
        let hidden: LinearLayer
        let output: LinearLayer
        let revived: RevivedRowsLayer?
        var parameterCount: Int {
            norm.parameterCount + down.parameterCount + hidden.parameterCount + output.parameterCount
        }
    }

    let paths: [(sense: SenseName, path: HeadPath)]
    /// The running centre [6, 192] (buffer centre_mean, frozen at inference).
    let centre: MLXArray
    /// The memory's constant [6, 192] subtracted first; nil without a persistent memory.
    let constant: MLXArray?
    /// The output gain 1 / sqrt(hidden width).
    let gain: Float

    /// The path stored under `source` (the released file seen through
    /// `scoped("heads_all_slot_path.")`) for `senses`, with the memory's `constant` [6, 192] or nil and
    /// the revived units of each sense's net.0.
    public init(source: any MLXTensorSource, senses: [SenseName] = SenseName.goalScored, constant: MLXArray?,
                revived: [SenseName: RevivedRowsLayer] = [:], width: Int = TransformerCore.Released.width,
                slots: Int = SenseName.activeCount) throws {
        let rank = HeadsAllSlotPath.slotRank
        paths = try senses.map { (sense: SenseName) throws -> (sense: SenseName, path: HeadPath) in
            let prefix = "paths.\(sense.rawValue)."
            let norm = try LayerNormLayer(source: source, prefix: prefix + "norm.", width: width)
            let down = try LinearLayer(source: source, prefix: prefix + "down.", inputs: width, outputs: rank,
                                       hasBias: false)
            let hidden = try LinearLayer(source: source, prefix: prefix + "net.0.", inputs: slots * rank,
                                         outputs: width, hasBias: true)
            let output = try LinearLayer(source: source, prefix: prefix + "net.2.", inputs: width, outputs: width,
                                         hasBias: true)
            let path = HeadPath(norm: norm, down: down, hidden: hidden, output: output, revived: revived[sense])
            return (sense: sense, path: path)
        }
        centre = try source.tensor("centre_mean", shape: [slots, width])
        if let constant {
            guard constant.shape == [slots, width] else {
                throw ModelPortError.tensorShape(key: "all-slot path constant", expected: [slots, width],
                                                 found: constant.shape)
            }
        }
        self.constant = constant
        gain = 1 / Float(width).squareRoot()
    }

    /// The path of the released file: its weights under "heads_all_slot_path.", the constant
    /// persistent_memory.readout.bias [1152] reshaped to [6, 192], and the revived units of each
    /// net.0 with the centres of `centres`.
    public static func released(from weights: any MLXTensorSource,
                                centres: PredictionHeadCentres) throws -> HeadsAllSlotPath {
        let width = TransformerCore.Released.width
        let slots = SenseName.activeCount
        let bias = try weights.tensor("persistent_memory.readout.bias", shape: [slots * width])
        var revived: [SenseName: RevivedRowsLayer] = [:]
        for sense in SenseName.goalScored {
            let layer = "heads_all_slot_path__paths__\(sense.rawValue)__net__0"
            revived[sense] = try centres.revivedRows(layer: layer, from: weights, inputs: slots * slotRank,
                                                     outputs: width)
        }
        return try HeadsAllSlotPath(source: weights.scoped("heads_all_slot_path."),
                                    constant: bias.reshaped([slots, width]), revived: revived)
    }

    /// The parameter count (403,200 for the released path; the revived units not counted).
    public var parameterCount: Int { paths.reduce(0) { $0 + $1.path.parameterCount } }

    public func additions(lastStep: MLXArray) -> [SenseName: MLXArray] {
        let batch = lastStep.shape[0]
        var x = lastStep
        if let constant {
            x = x - constant
        }
        let centred = x - centre
        var out: [SenseName: MLXArray] = [:]
        for (sense, path) in paths {
            let reduced = path.down(path.norm(centred))
            let flat = reduced.reshaped([batch, reduced.shape[1] * reduced.shape[2]])
            var pre = path.hidden(flat)
            if let revived = path.revived {
                pre = revived.apply(input: flat, preActivation: pre)
            }
            out[sense] = path.output(NeuralOps.gelu(pre)) * gain
        }
        return out
    }
}

// MARK: - The heads

/// The prediction heads (PredictionHeads of prediction_heads.py:155-246): one head per sense except
/// motor, the residual base, and an optional addition.
public struct PredictionHeads {
    /// Parameters of the component prediction_heads: 1,114,560.
    public static let parameterCount = 1_114_560

    /// The senses with a head, in the receipt's sequence.
    public let senses: [SenseName]
    let heads: [SenseName: PredictionHead]
    /// True when a head predicts the change and its sense's raw token is added (residual_prediction on).
    public let residual: Bool
    /// The all-slot path; nil adds nothing.
    public let addition: (any PredictionAdding)?

    /// Heads over given parts.
    public init(heads: [SenseName: PredictionHead], senses: [SenseName] = SenseName.goalScored, residual: Bool = true,
                addition: (any PredictionAdding)? = nil) {
        self.heads = heads
        self.senses = senses
        self.residual = residual
        self.addition = addition
    }

    /// The heads stored under `source` (the released file seen through `scoped("prediction_heads.")`),
    /// with the revived units of each sense's first Linear from `revived`.
    public init(source: any MLXTensorSource, senses: [SenseName] = SenseName.goalScored, residual: Bool = true,
                addition: (any PredictionAdding)? = nil, revived: [SenseName: RevivedRowsLayer] = [:]) throws {
        var built: [SenseName: PredictionHead] = [:]
        for sense in senses {
            built[sense] = try PredictionHead(source: source.scoped("heads.\(sense.rawValue)."), revived: revived[sense])
        }
        self.init(heads: built, senses: senses, residual: residual, addition: addition)
    }

    /// The released heads: weights under "prediction_heads.", the revived units of the first Linear
    /// where the weights hold a mask (vision, tactile, vestibular), and `addition`.
    public static func released(from weights: any MLXTensorSource, centres: PredictionHeadCentres,
                                addition: (any PredictionAdding)?) throws -> PredictionHeads {
        let width = TransformerCore.Released.width
        var revived: [SenseName: RevivedRowsLayer] = [:]
        for sense in SenseName.goalScored {
            let layer = "prediction_heads__heads__\(sense.rawValue)__mlp__0"
            revived[sense] = try centres.revivedRows(layer: layer, from: weights, inputs: width, outputs: width)
        }
        return try PredictionHeads(source: weights.scoped("prediction_heads."), addition: addition, revived: revived)
    }

    /// The parameter count of the heads (1,114,560 for the released heads; the addition not counted).
    public var parameterCount: Int { heads.values.reduce(0) { $0 + $1.parameterCount } }

    /// The predicted next token per sense [B, 192]. `current` holds this model step's raw tokens
    /// [B, 192] per sense; with `residual` a sense without one throws `ModelPortError.missingInput`
    /// (prediction_heads.py:203-223 raises in the same case).
    public func callAsFunction(lastStep: MLXArray, current: [SenseName: MLXArray]) throws -> [SenseName: MLXArray] {
        let extra = addition?.additions(lastStep: lastStep) ?? [:]
        var predictions: [SenseName: MLXArray] = [:]
        for sense in senses {
            guard let head = heads[sense] else {
                throw ModelPortError.missingInput(sense)
            }
            var prediction = head(lastStep)
            if residual {
                guard let base = current[sense] else {
                    throw ModelPortError.missingInput(sense)
                }
                prediction = prediction + base
            }
            if let add = extra[sense] {
                prediction = prediction + add
            }
            predictions[sense] = prediction
        }
        return predictions
    }
}
