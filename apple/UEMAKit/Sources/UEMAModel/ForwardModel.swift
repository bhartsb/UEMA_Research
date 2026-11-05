// ForwardModel.swift: one model step of the released model's sensing and prediction path in MLX
// Swift: the six encoders with the shared output LayerNorm, the context buffer of 127 previous
// model steps with the motor-slot back-fill, the transformer and the prediction heads (work item
// WI-35 of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-9 and
// REQ-B-10). Nothing in this file was compiled on the machine that wrote it (no Swift toolchain on
// Windows).
//
// INPUT per model step: one EncoderInput per active sense (UEMATokenizer, WI-30): vision patches
//   [N, 768] with angles [N, 4]; tactile [32, 19]; audio [256, 2] with spatial [2]; proprioception
//   [312]; motor [26, 19] with zero command columns; vestibular [28]; and the command executed on
//   the previous model step (scale `.centimetres`; nil on the first model step after a reset).
// OPERATION (mimo_bridge/lem_model.py LemLite.step, read November 4, 2026; line numbers of the copy
//   under D:/UEMA_Research/Source/Python):
//   1. Back-fill (lem_model.py:2629-2656): when a previous executed command exists, the previous
//      model step's motor slot in the context buffer (index 762 - 6 + 4 = 760, the motor sense
//      being the fifth active sense) is replaced by LayerNorm(motor encoder([previous achieved 10,
//      command with positions / 200 cm])) plus the motor row of the modality embedding.
//   2. Encode (encoder_registry.py:129-211): each encoder's pooled vector [1, 1, 192] (the tactile
//      and motor pools with the liveness repair's all-element path added, TactileEncoder.swift step
//      6), then the shared output LayerNorm without affine parameters (encoder_registry.py:58-82,
//      178-179): the raw tokens.
//   3. On the first model step after a reset the forward model reads the raw tokens; the
//      top-down gate's blend (args.json top_down_gate on; td_gates, outside the manifest's inference
//      set) mixes up to half of the previous prediction into the tokens on later model steps
//      (lem_model.py around line 2900: tokens[m] + g (prediction - tokens[m])). It is not ported in
//      this work item, so from the second model step on the tokens this port feeds differ from the
//      Python ones by that blend, bounded by g <= 0.5 (the fixtures cover the first model step).
//   4. Memory bias: the persistent memory's bias [1, 6, 192] (args.json persistent_memory on;
//      component persistent_memory, outside the inference set; the fixture's forward/memory_bias)
//      comes from an injected MemoryBiasProviding; NoMemoryBias gives none.
//   5. Transformer (TransformerCore.swift) on the 6 raw tokens and the context buffer.
//   6. Prediction heads (PredictionHeads.swift) on the last model step with the raw tokens as the
//      residual base (lem_model.py:2983-2986), with the liveness repair's revived units and the
//      all-slot path; the revived units' centres come from Resources/prediction_head_centres.json.
//   7. The context buffer drops its oldest model step and appends this model step's frame (raw
//      tokens plus the modality embedding) (lem_model.py:4090).
// OUTPUT: a ForwardPass: the six raw tokens, the five predicted next tokens, the latents
//   [1, 768, 192] and the last step [1, 6, 192] the motor controller and wellness read (WI-36), and
//   the achieved handle state [1, 26, 10] of this model step.
// WHAT IS NOT HERE: the motor controller, wellness and the command (WI-36), so ForwardModel does not
//   conform to UEMACore's ForwardModeling, whose ModelOutput carries a command and a wellness score;
//   the type that does joins this one with WI-36's MotorController and wellness scorer.
// FIRST BUILD ON A MAC: see TransformerCore.swift, steps 1 to 4.

import Foundation
import MLX
import UEMACore

// MARK: - One sense's encoder

/// One active sense's encoder: an EncoderInput into the pooled vector [1, 1, 192] before the shared
/// output LayerNorm (SenseEncoder in the specification's section 5).
public protocol SenseEncoding {
    /// The sense this encoder serves.
    var sense: SenseName { get }
    /// The encoder's parameter count.
    var parameterCount: Int { get }
    /// The pooled vector [1, 1, 192] of one model step's input; a wrong sense or shape throws.
    func encode(_ input: EncoderInput) throws -> MLXArray
}

// MARK: - The six encoders and the shared output LayerNorm

/// The six active senses' encoders and the registry's shared output LayerNorm (EncoderRegistry of
/// encoder_registry.py:27-211).
public struct EncoderSet {
    /// The encoders by sense, one per active sense.
    public let encoders: [SenseName: any SenseEncoding]
    /// The motor encoder, kept typed for the back-fill and the planner's gradient (WI-39).
    public let motor: MotorEncoder
    /// LayerNorm(192) without affine parameters, eps 1e-5 (encoder_registry.py:81).
    public let outputNorm = LayerNormLayer(weight: nil, bias: nil)

    /// A set over given encoders; every active sense must have exactly one and `motor` serves motor.
    public init(motor: MotorEncoder, others: [any SenseEncoding]) throws {
        var bySense: [SenseName: any SenseEncoding] = [.motor: motor]
        for encoder in others {
            bySense[encoder.sense] = encoder
        }
        for sense in SenseName.active where bySense[sense] == nil {
            throw ModelPortError.missingInput(sense)
        }
        self.encoders = bySense
        self.motor = motor
    }

    /// The released encoders read from the weights (keys "encoder_registry.encoders.<sense>.").
    public static func released(from weights: any MLXTensorSource,
                                motorAdjustment: (any PooledVectorAdjusting)? = nil) throws -> EncoderSet {
        func scope(_ sense: SenseName) -> ScopedMLXTensorSource {
            weights.scoped("encoder_registry.encoders.\(sense.rawValue).")
        }
        // the liveness repair's all-element paths beside the motor and tactile pools (TactileEncoder.swift step 6)
        let motorPath = try AllElementPath.released(sense: .motor, elements: MotorEncoder.handles, from: weights)
        let tactilePath = try AllElementPath.released(sense: .tactile, elements: TactileEncoder.points, from: weights)
        let motor = try MotorEncoder(source: scope(.motor), poolPath: motorPath, adjustment: motorAdjustment)
        let vision = try VisionEncoder(source: scope(.vision))
        let tactile = try TactileEncoder(source: scope(.tactile), poolPath: tactilePath)
        let audio = try AudioEncoder(source: scope(.audio))
        let proprioception = try ProprioceptionEncoder(source: scope(.proprioception))
        let vestibular = try VestibularEncoder(source: scope(.vestibular))
        let others: [any SenseEncoding] = [vision, tactile, audio, proprioception, vestibular]
        return try EncoderSet(motor: motor, others: others)
    }

    /// The parameter count of the six encoders: 4,999,680 for the released ones (2,106,048 +
    /// 1,074,816 + 450,432 + 171,648 + 1,079,808 + 117,888).
    public var parameterCount: Int { encoders.values.reduce(0) { $0 + $1.parameterCount } }

    /// The pooled vector (before the shared norm) of each active sense, [1, 1, 192] each.
    public func pooled(_ inputs: [SenseName: EncoderInput]) throws -> [SenseName: MLXArray] {
        var out: [SenseName: MLXArray] = [:]
        for sense in SenseName.active {
            guard let input = inputs[sense], let encoder = encoders[sense] else {
                throw ModelPortError.missingInput(sense)
            }
            out[sense] = try encoder.encode(input)
        }
        return out
    }

    /// The raw token of a pooled vector: the shared output LayerNorm.
    public func token(_ pooled: MLXArray) -> MLXArray {
        outputNorm(pooled)
    }

    /// The raw motor token [B, 1, 192] of motor rows [B, 26, 19] (imagined_motor_token of
    /// encoder_registry.py:302-320: through the registry, so it gets the shared LayerNorm).
    public func motorToken(rows: MLXArray) -> MLXArray {
        outputNorm(motor(rows))
    }
}

// MARK: - The persistent memory's bias

/// The source of the persistent memory's bias [1, 6, 192] added to the last model step's tokens
/// after the transformer's final norm (forward_model.py:415-419).
public protocol MemoryBiasProviding {
    /// The bias for this model step given its raw tokens [1, 6, 192]; nil adds nothing.
    func memoryBias(tokens: MLXArray, at step: ModelStep) throws -> MLXArray?
}

/// No persistent memory: no bias.
public struct NoMemoryBias: MemoryBiasProviding {
    public init() {}

    public func memoryBias(tokens: MLXArray, at step: ModelStep) throws -> MLXArray? {
        nil
    }
}

/// A bias fixed in advance (a fixture's forward/memory_bias, or a measured constant).
public struct FixedMemoryBias: MemoryBiasProviding {
    /// The bias [1, 6, 192].
    public let bias: MLXArray

    /// A provider that returns `bias` on every model step.
    public init(bias: MLXArray) {
        self.bias = bias
    }

    public func memoryBias(tokens: MLXArray, at step: ModelStep) throws -> MLXArray? {
        bias
    }
}

// MARK: - What one model step produced

/// The outputs of one model step of the sensing and prediction path.
public struct ForwardPass {
    /// The model step these outputs belong to.
    public let step: ModelStep
    /// The six raw tokens (after the shared LayerNorm), one per active sense.
    public let tokens: TokenSet
    /// The predicted next token of each sense with a prediction head (five; motor has none).
    public let predictedTokens: TokenSet
    /// The six raw tokens as one array [1, 6, 192] in the active sense sequence.
    public let tokenArray: MLXArray
    /// The pooled vectors before the shared LayerNorm, [1, 1, 192] per sense.
    public let pooled: [SenseName: MLXArray]
    /// The predicted next tokens as arrays [1, 192] per sense.
    public let predictedArrays: [SenseName: MLXArray]
    /// The transformer's output (latents [1, 768, 192], last step [1, 6, 192], context update).
    public let transformer: TransformerOutput
    /// The memory bias the transformer added, or nil.
    public let memoryBias: MLXArray?
    /// The context buffer the transformer read, [1, 762, 192].
    public let contextIn: MLXArray
    /// This model step's achieved handle state [1, 26, 10] (the motor input's first 10 numbers).
    public let achieved: MLXArray
}

// MARK: - The context buffer

/// The context buffer of the previous 127 model steps' frames, [1, 762, 192] (lem_model.py:2297
/// reset_context: zeros [B, (ctx_len - 1) x n_mod, d_model]), and its two edits: the motor-slot
/// back-fill and the shift that appends a frame.
public struct ContextBuffer {
    /// Model steps in the transformer's context: 128.
    public let contextLength: Int
    /// Tokens per model step: 6.
    public let senseCount: Int
    /// Token width: 192.
    public let width: Int
    /// The frames, [1, (contextLength - 1) x senseCount, width].
    public private(set) var frames: MLXArray
    /// Model steps appended since the last reset, at most contextLength - 1.
    public private(set) var filledSteps: Int = 0

    /// An all-zero buffer.
    public init(contextLength: Int = TransformerCore.Released.contextLength,
                senseCount: Int = TransformerCore.Released.senseCount,
                width: Int = TransformerCore.Released.width) {
        self.contextLength = contextLength
        self.senseCount = senseCount
        self.width = width
        self.frames = zeros([1, (contextLength - 1) * senseCount, width], type: Float.self)
    }

    /// Back to zeros.
    public mutating func reset() {
        frames = zeros([1, (contextLength - 1) * senseCount, width], type: Float.self)
        filledSteps = 0
    }

    /// Replaces the newest model step's token of the sense at `senseIndex` with `frame` [1, 192]
    /// (lem_model.py:2656: context_buffer[:, -(n_mod - mi), :] = z_bf).
    public mutating func replaceNewest(senseIndex: Int, with frame: MLXArray) {
        let total = frames.shape[1]
        let index = total - senseCount + senseIndex
        let row = frame.reshaped([1, 1, width])
        if index + 1 < total {
            frames = concatenated([frames[0..., 0 ..< index], row, frames[0..., (index + 1)...]], axis: 1)
        } else {
            frames = concatenated([frames[0..., 0 ..< index], row], axis: 1)
        }
    }

    /// Drops the oldest model step and appends `frame` [1, 6, 192] (lem_model.py:4090).
    public mutating func append(_ frame: MLXArray) {
        frames = concatenated([frames[0..., senseCount...], frame], axis: 1)
        filledSteps = min(filledSteps + 1, contextLength - 1)
    }
}

// MARK: - The forward model

/// One model step of the sensing and prediction path, with the context buffer as state.
public final class ForwardModel {
    /// The six encoders and the shared LayerNorm.
    public let encoders: EncoderSet
    /// The transformer.
    public let core: TransformerCore
    /// The prediction heads (with the all-slot path when injected).
    public let heads: PredictionHeads
    /// The persistent memory's bias source.
    public let memoryBias: any MemoryBiasProviding
    /// The context buffer.
    public private(set) var context: ContextBuffer
    private var lastAchieved: MLXArray?

    /// A forward model over given parts.
    public init(encoders: EncoderSet, core: TransformerCore, heads: PredictionHeads,
                memoryBias: any MemoryBiasProviding = NoMemoryBias()) {
        self.encoders = encoders
        self.core = core
        self.heads = heads
        self.memoryBias = memoryBias
        self.context = ContextBuffer()
    }

    /// The released model read from the weights: the six encoders (with the liveness repair's pool
    /// paths), forward_model, prediction_heads with their revived units and, when `withAllSlotPath`,
    /// the heads' all-slot path with its constant from persistent_memory.readout.bias (the released
    /// run's flags heads_all_slot_path on and liveness_repair on). `centres` defaults to the bundled
    /// Resources/prediction_head_centres.json.
    public static func released(from weights: any MLXTensorSource, centres: PredictionHeadCentres? = nil,
                                withAllSlotPath: Bool = true,
                                memoryBias: any MemoryBiasProviding = NoMemoryBias(),
                                motorAdjustment: (any PooledVectorAdjusting)? = nil) throws -> ForwardModel {
        let centreTable = try centres ?? PredictionHeadCentres.bundled()
        let encoders = try EncoderSet.released(from: weights, motorAdjustment: motorAdjustment)
        let core = try TransformerCore(source: weights.scoped("forward_model."))
        var addition: (any PredictionAdding)? = nil
        if withAllSlotPath {
            addition = try HeadsAllSlotPath.released(from: weights, centres: centreTable)
        }
        let heads = try PredictionHeads.released(from: weights, centres: centreTable, addition: addition)
        return ForwardModel(encoders: encoders, core: core, heads: heads, memoryBias: memoryBias)
    }

    /// The model steps the context holds so far, 0 to 127.
    public var filledContextSteps: Int { context.filledSteps }

    /// Empties the context and forgets the last achieved state (reset_context, lem_model.py:2297).
    public func reset() {
        context.reset()
        lastAchieved = nil
    }

    /// One model step: back-fill, encode, transformer, heads, context shift (this file's header).
    public func step(inputs: [SenseName: EncoderInput], previousExecutedCommand: Command?,
                     at step: ModelStep) throws -> ForwardPass {
        guard let motorIndex = SenseName.motor.tokenIndex else {
            throw ModelPortError.missingInput(.motor)
        }
        // 1. Back-fill the previous model step's motor slot.
        if let command = previousExecutedCommand, let previous = lastAchieved {
            guard command.scale == .centimetres else {
                throw ModelPortError.wrongCommandScale(expected: .centimetres, found: command.scale)
            }
            let commandRows = MLXArray(command.normalized().values, [1, Command.handleCount, Command.numbersPerHandle])
            let rows = MotorEncoder.rows(achieved: previous, command: commandRows)
            let token = encoders.motorToken(rows: rows).reshaped([1, TransformerCore.Released.width])
            let frame = token + core.modalityEmbedding[motorIndex]
            context.replaceNewest(senseIndex: motorIndex, with: frame)
        }
        // 2. Encode the six senses.
        let pooled = try encoders.pooled(inputs)
        var tokenRows: [MLXArray] = []
        var raw: [SenseName: MLXArray] = [:]
        for sense in SenseName.active {
            guard let vector = pooled[sense] else {
                throw ModelPortError.missingInput(sense)
            }
            let token = encoders.token(vector).reshaped([1, TransformerCore.Released.width])
            raw[sense] = token
            tokenRows.append(token)
        }
        let tokenArray = stacked(tokenRows, axis: 1)
        // 4. Memory bias; 5. transformer.
        let bias = try memoryBias.memoryBias(tokens: tokenArray, at: step)
        let contextIn = context.frames
        let output = try core(tokens: tokenArray, context: contextIn, memoryBias: bias)
        // 6. Prediction heads with the raw tokens as the residual base.
        let predicted = try heads(lastStep: output.lastStep, current: raw)
        // 7. Shift the context; keep this model step's achieved state for the next back-fill.
        context.append(output.contextUpdate)
        guard let motorInput = inputs[.motor] else {
            throw ModelPortError.missingInput(.motor)
        }
        let achieved = try NeuralOps.batched(motorInput.primary,
                                             expected: [MotorEncoder.handles, MotorEncoder.numbersPerRow],
                                             what: "motor rows")[0..., 0..., 0 ..< MotorEncoder.achievedNumbers]
        lastAchieved = achieved
        return ForwardPass(step: step,
                           tokens: try ForwardModel.tokenSet(raw),
                           predictedTokens: try ForwardModel.tokenSet(predicted),
                           tokenArray: tokenArray,
                           pooled: pooled,
                           predictedArrays: predicted,
                           transformer: output,
                           memoryBias: bias,
                           contextIn: contextIn,
                           achieved: achieved)
    }

    /// Arrays [1, 192] per sense as UEMACore Tokens; a non-finite number throws.
    static func tokenSet(_ arrays: [SenseName: MLXArray]) throws -> TokenSet {
        var out: TokenSet = [:]
        for (sense, array) in arrays {
            let values = NeuralOps.floats(array)
            guard values.allSatisfy({ $0.isFinite }) else {
                throw ModelPortError.nonFinite(what: "the token of \(sense.rawValue)")
            }
            out[sense] = try Token(values: values)
        }
        return out
    }
}
