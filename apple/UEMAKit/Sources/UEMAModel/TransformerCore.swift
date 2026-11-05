// TransformerCore.swift: the shared layers of the released model in MLX Swift (Linear, LayerNorm,
// GELU, PyTorch-layout multi-head attention, the pre-norm attention block) and the forward model's
// transformer of 4 three-stage blocks over the 768-token context (work item WI-35 of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-10 of
// docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html). Nothing in this file was compiled on the
// machine that wrote it (Windows 11, no Swift toolchain, no Xcode: plan section 3, fact 1).
//
// WHAT THIS FILE DOES.
// INPUT: frozen float32 weight arrays read key by key from model.safetensors through an MLXTensorSource
//   (LoadedWeights of SafetensorsLoader.swift, WI-31, conforms below), under the exact PyTorch
//   state_dict names the weight manifest lists (Resources/weight_manifest.json, WI-12).
// OPERATION: the arithmetic of the PyTorch modules, each written as MLX array operations:
//   Linear y = x W^T + b (torch.nn.Linear); LayerNorm over the last axis, eps 1e-5, biased
//   variance (torch.nn.LayerNorm); GELU exact, 0.5 x (1 + erf(x / sqrt 2)) (torch.nn.GELU default
//   approximate='none'); multi-head attention with in_proj_weight [3D, D] split into the query,
//   key and value rows in that sequence, scores q k^T / sqrt(D / heads), softmax over the keys,
//   out_proj (torch.nn.MultiheadAttention, batch_first=True). A boolean PyTorch attn_mask entry
//   True (masked) is an additive -infinity here.
// OUTPUT: for TransformerCore, latents [B, 768, 192], the last model step's 6 tokens [B, 6, 192]
//   and the context update [B, 6, 192] (the current tokens plus the modality embedding, before the
//   blocks), the three values ForwardModel.forward returns (forward_model.py:414-426).
//
// WHY ARRAYS, NOT MLXNN MODULE TREES. The specification (REQ-B-10) asks for "MLXNN modules with the
// same tensor names". The released names contain sparse Sequential indices ("mlp.0", "mlp.1",
// "mlp.4", "mlp.12"; "spectral_conv.0", "spectral_conv.3"), which ModuleParameters.unflattened
// turns into array levels sized to the largest index plus one (SafetensorsLoader.swift header,
// Source/MLX/Nested.swift), so a Module tree would need placeholder modules at the empty indices.
// Each layer here instead reads its arrays by the full key the manifest names, through
// MLXTensorSource.tensor(_:shape:), which refuses a missing key or a wrong shape by name: the
// mapping from PyTorch name to Swift array is one to one and checked at construction. The model
// is frozen (REQ-B-17), so no optimizer needs a Module parameter tree, and the planner's gradient
// with respect to the command (WI-39) is a gradient of a function of MLXArrays, which MLX's
// grad / valueAndGrad transforms take whatever holds the weights.
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave:
// - Source/Python/lem/model/forward_model.py:34-184 (ThreeStageBlock): stage 1 cross-modal
//   attention per model step over the 6 sense tokens (cm_norm, cm_attn, residual); stage 2
//   causal-temporal attention per sense over the 128 model steps with
//   causal = torch.triu(ones(128, 128), diagonal=1) as attn_mask (ct_norm, ct_attn, residual);
//   stage 3 the bridge, full attention over all 768 tokens with no mask (cmb_norm, cmb_attn,
//   residual); then ffn_norm and ffn = Linear(192, 768), GELU, Dropout, Linear(768, 192), Dropout,
//   residual. Layout: "[t0_m0, t0_m1, ..., t0_mN, t1_m0, ...] i.e. modalities interleaved within
//   each timestep" (forward_model.py:102-103).
// - forward_model.py:298-426 (ForwardModel.forward): current_frame = tokens [B, 6, D] plus the
//   modality embedding (row i for the i-th active sense); sequence = cat(context_buffer
//   [B, 762, D], current) along the sequence; plus the temporal embedding of model step t added to
//   all 6 tokens of step t; the blocks; final_norm; then, when given, memory_bias [B, 6, D] added
//   to the last 6 tokens only (forward_model.py:415-419); last_step = the last 6 tokens.
// - lem/model/attention_cap.py:29 and args.json "attention_cap off": CappedMultiheadAttention runs
//   nn.MultiheadAttention's own forward while it carries no bound, so every attention module of
//   the released run is plain multi-head attention.
// - lem/model/encoders/swin_patch.py:70-118, cross_patch.py:90-114: the pre-norm residual
//   attention block (norm, attention, residual; norm, feed-forward, residual) that
//   PreNormAttentionBlock below implements for the encoders.
// - model_receipt.json of checkpoint 494,025, forward_model: {"_seq_length": 768,
//   "_n_modalities": 6, "n_layers": 4, "n_heads": 8, "ffn_expansion": 4}; component_params
//   forward_model 2,994,432.
// - weight_manifest.json keys forward_model.blocks.<0-3>.{cm,ct,cmb}_attn.in_proj_weight [576, 192],
//   in_proj_bias [576], out_proj.weight [192, 192], out_proj.bias [192], the norms' weight and
//   bias [192], ffn.0.weight [768, 192], ffn.0.bias [768], ffn.3.weight [192, 768], ffn.3.bias
//   [192], causal_mask [128, 128] BOOL (a buffer, rebuilt here from its rule), final_norm.weight
//   and bias [192], modality_embedding.weight [6, 192], temporal_embedding.weight [128, 192].
//   Parameter arithmetic per block: 3 x (110,592 + 576 + 36,864 + 192 + 384) + 384
//   + 147,456 + 768 + 147,456 + 192 = 742,080; 4 blocks 2,968,320, plus final_norm 384,
//   modality 1,152 and temporal 24,576 = 2,994,432, the receipt's count.
// - mlx-swift source files at https://raw.githubusercontent.com/ml-explore/mlx-swift/main/Source/MLX/
//   read November 4, 2026 (the documentation pages at
//   https://swiftpackageindex.com/ml-explore/mlx-swift/documentation/mlx name the same symbols):
//   Ops.swift "public func erf(_ array: MLXArray, stream: StreamOrDevice = .default) -> MLXArray",
//   "public func softmax(_ array: MLXArray, axis: Int, precise: Bool = false, ...)",
//   "public func concatenated(_ arrays: some Collection<MLXArray>, axis: Int = 0, ...)",
//   "public func stacked(_ arrays: some Collection<MLXArray>, axis: Int = 0, ...)",
//   "public func broadcast(_ array: MLXArray, to shape: some Collection<Int>, ...)",
//   "public func roll(_ a: MLXArray, shift: Int, axis: Int, ...)",
//   "public func conv1d(_ array: MLXArray, _ weight: MLXArray, stride: Int = 1, padding: Int = 0,
//   dilation: Int = 1, groups: Int = 1, ...)", "public func sigmoid(_ array: MLXArray, ...)";
//   MLXArray+Ops.swift methods matmul(_:), mean(axis:keepDims:), sum(axis:keepDims:), square(),
//   sqrt(), rsqrt(), reshaped(_ newShape: some Collection<Int>), transposed(_ axes: Int...), T,
//   split(parts:axis:), expandedDimensions(axis:), and the operators + - * / between arrays and
//   with "some ScalarOrArray"; Factory.swift "public func zeros(_ shape: some Collection<Int>,
//   type: (some HasDType).Type = Float.self, ...)"; MLXArray+Init.swift
//   "public convenience init<T: HasDType>(_ value: [T], _ shape: (some Collection<Int>)? = ...)";
//   MLXArray+Indexing.swift: Int, Range<Int>, PartialRangeFrom<Int> and the rest conform to
//   MLXArrayIndex, read through "public subscript(indices: any MLXArrayIndex..., ...)";
//   MLXArray.swift "final public var shape: [Int]", "final public func item<T: HasDType>(_ type:
//   T.Type) -> T", "final public func asType(_ type: DType, ...)"; Ops.swift "public func which(_
//   condition: MLXArray, _ a: some ScalarOrArray, _ b: some ScalarOrArray, ...)" (where). Not read
//   from source and used as other UEMAModel files use it: MLXArray.asArray(_:) (SafetensorsLoader
//   tests call weight.asArray(Float.self)).
//
// SOLID. S: one layer kind per type (LinearLayer, LayerNormLayer, MultiheadAttentionLayer,
// PreNormAttentionBlock, ThreeStageBlock); TransformerCore owns the embeddings and the block
// sequence and nothing else. D: every layer reads its weights through the MLXTensorSource protocol,
// so tests inject an MLXDictionaryTensorSource built from the manifest's shapes and the app injects
// LoadedWeights.
//
// FIRST BUILD ON A MAC (this work item's acceptance: "a forward pass on a WI-34 state on the Metal
// device, no NaN, matching the fixture's predicted tokens within a tolerance recorded in section 9").
//  1. apple/README.md steps 1 to 3 have passed (weights fetched, MuJoCo XCFramework built,
//     `swift package resolve` exit 0) and SafetensorsLoaderTests pass (WI-31).
//  2. `cd apple/UEMAKit && xcodebuild -scheme UEMAKit build -destination 'platform=macOS'`
//     : exit 0. Constructs to check first if it fails, each used on purpose: the variadic
//     `transposed(0, 2, 1, 3)`; the subscripts `x[0..., 0 ..< n]` and `x[0..., n...]`;
//     `split(parts: 3, axis: 0)` returning [MLXArray]; `x * Float(...)` (MLXArray times
//     ScalarOrArray); `-Float.infinity` inside an MLXArray mask; `which(mask, a, b)` with a BOOL
//     mask; the extension making the struct LoadedWeights conform to MLXTensorSource (WI-36's
//     MotorController.swift declares a different protocol, TensorSource, returning ShapedArray; the
//     two coexist under different names). The resource Resources/prediction_head_centres.json must
//     be in the module bundle (Package.swift copies the whole Resources folder).
//  3. `xcodebuild -scheme UEMAKit test -destination 'platform=macOS'
//     -only-testing:UEMAModelTests/ModelPassTests` : exit 0. Without model.safetensors the
//     shape tests run (each component reads exactly its manifest keys and parameter count; every
//     output has its shape and is finite); with it (found at the manifest's source.path under the
//     repository root or at UEMA_WEIGHTS_PATH) the three fixture states run and print, per sense,
//     the largest absolute difference of the token and of the predicted token against
//     Fixtures/model_pass/<state>.safetensors. The pass line is ModelPassTolerance below
//     (1e-3 token units; a real token has a root mean square of 1.0): write each printed number
//     into the plan's section 9 row of WI-35. Before compiling, a float64 numpy copy of this
//     port's arithmetic reproduced the fixtures within 1.4e-6 (PredictionHeads.swift header), so a
//     difference above the line points at a Swift or MLX construct, not at the layer sequence.
//  4. Record each step's command, exit code and clock time (AM/PM Pacific) in
//     apple/FIRST_BUILD_RECORD.md (WI-41).

import Foundation
import MLX
import UEMACore

// MARK: - Errors

/// The errors of the model port: a tensor the weights lack or hold in another shape, an input of
/// the wrong shape, an active sense without an input, or a non-finite output.
public enum ModelPortError: Error, Equatable, CustomStringConvertible {
    /// The weights hold no tensor named `key`.
    case missingTensor(key: String)
    /// The tensor `key` has the shape `found` where the layer reads `expected`.
    case tensorShape(key: String, expected: [Int], found: [Int])
    /// The input `what` has the shape `found` where `expected` is read (-1 marks a free extent).
    case inputShape(what: String, expected: [Int], found: [Int])
    /// The model step's inputs lack this active sense.
    case missingInput(SenseName)
    /// A command of the wrong scale was handed to the context back-fill.
    case wrongCommandScale(expected: CommandScale, found: CommandScale)
    /// An output named `what` holds NaN or infinity.
    case nonFinite(what: String)

    public var description: String {
        switch self {
        case let .missingTensor(key):
            return "the weights hold no tensor named \(key)"
        case let .tensorShape(key, expected, found):
            return "tensor \(key) has shape \(found), the layer reads \(expected)"
        case let .inputShape(what, expected, found):
            return "input \(what) has shape \(found), expected \(expected)"
        case let .missingInput(sense):
            return "no encoder input for the active sense \(sense.rawValue)"
        case let .wrongCommandScale(expected, found):
            return "a command at scale \(found.rawValue) was given where \(expected.rawValue) is read"
        case let .nonFinite(what):
            return "\(what) holds a non-finite number"
        }
    }
}

// MARK: - The fixture pass line

/// The declared pass line of the fixture comparison (plan row WI-35, "matching the fixture's
/// predicted tokens within a tolerance recorded in section 9"), set before any Swift run.
/// QUANTITY: the largest absolute difference, over the 192 numbers of a token, between the Swift
/// token (or predicted token) and the fixture's, in token units; a real token has a root mean square
/// of 1.0 (encoder_registry.py:58-82: the shared LayerNorm), so 0 is identical and 1 is a difference
/// the size of a typical number. LINE: 1e-3. WHY: (a) the float64 numpy copy of this port reproduced
/// the float32 fixtures within 1.4e-6 on all three states (November 4, 2026), so the arithmetic is
/// the Python's and what remains is float32 rounding; (b) float32 has a unit roundoff of 6e-8, the
/// longest dot products here have 1,160 terms (the vision pool) and 768 (the feed-forward layers),
/// and 84 layers lie in sequence from a vision patch to a predicted token (1 + 4 x 7 + 3 in the
/// vision encoder, 4 x 12 in the transformer, 4 in a head), so the worst-case accumulated relative
/// rounding is of order 1,160 x 6e-8 x 84 = 5.8e-3, and 1e-3 sits below that worst case because the
/// rounding errors of independent terms partly cancel; (c) the specification's own band (REQ-B-10: "Swift stays within that band of the cpu
/// result", the cpu-versus-cuda band of WI-46) is unmeasured today. A first-build difference above
/// 1e-3 reads FAILED in section 9, never "recorded".
/// The float32 precision of MLX's Metal matrix products. mlx (the C++ core under mlx-swift) routes
/// a float32 matmul through the GPU's tensor units when `MLX_ENABLE_TF32` is 1, its default
/// (mlx/utils.h `enable_tf32_ = get_var("MLX_ENABLE_TF32", 1)`; mlx/backend/metal/matmul.cpp
/// `(env::enable_tf32() || a.dtype() != float32)` gating the nax kernels; read in mlx-swift 0.32.3
/// on November 4, 2026). Those units round the products to about 10 mantissa bits. The first Mac
/// build (Apple M5, mlx-swift 0.33.0) measured the result: pooled vectors of the three encoders
/// with real matrix products (vision, tactile, motor) 2e-3 to 2.7e-3 relative off the fixtures,
/// the batch-1 MLP encoders within 1e-5, while the same arithmetic in float32 numpy is within
/// 1.3e-6 on every token. The variable is read once, at the first matmul of the process, so
/// `enforce()` runs before the weights are read.
public enum MLXFloat32Precision {
    /// Sets MLX_ENABLE_TF32=0 for this process unless it is already set; returns the value in
    /// force. Call before the first MLX operation.
    @discardableResult
    public static func enforce() -> String {
        if let existing = ProcessInfo.processInfo.environment["MLX_ENABLE_TF32"] {
            return existing
        }
        setenv("MLX_ENABLE_TF32", "0", 1)
        return "0"
    }
}

public enum ModelPassTolerance {
    /// The pass line for tokens and predicted tokens: 1e-3 token units.
    public static let maxAbsoluteDifference: Float = 1e-3
}

// MARK: - Where the weights come from

/// Read access to the released weights by their safetensors key
/// ("forward_model.blocks.0.cm_attn.in_proj_weight"). LoadedWeights (WI-31) conforms; tests inject
/// an MLXDictionaryTensorSource.
public protocol MLXTensorSource {
    /// The array stored under `key`; throws `ModelPortError.missingTensor` when there is none.
    func tensor(_ key: String) throws -> MLXArray
}

extension MLXTensorSource {
    /// The array stored under `key`, refused with `ModelPortError.tensorShape` unless its shape is
    /// `shape`.
    public func tensor(_ key: String, shape: [Int]) throws -> MLXArray {
        let array = try tensor(key)
        guard array.shape == shape else {
            throw ModelPortError.tensorShape(key: key, expected: shape, found: array.shape)
        }
        return array
    }

    /// True when a tensor named `key` exists.
    public func hasTensor(_ key: String) -> Bool {
        (try? tensor(key)) != nil
    }

    /// This source seen through a key prefix: `scoped("forward_model.").tensor("final_norm.bias")`
    /// reads "forward_model.final_norm.bias".
    public func scoped(_ prefix: String) -> ScopedMLXTensorSource {
        ScopedMLXTensorSource(base: self, prefix: prefix)
    }
}

/// An MLXTensorSource that reads `prefix + key` from another source.
public struct ScopedMLXTensorSource: MLXTensorSource {
    /// The source read from.
    public let base: any MLXTensorSource
    /// The text put in front of every key.
    public let prefix: String

    /// A view of `base` under `prefix`.
    public init(base: any MLXTensorSource, prefix: String) {
        self.base = base
        self.prefix = prefix
    }

    public func tensor(_ key: String) throws -> MLXArray {
        try base.tensor(prefix + key)
    }
}

/// An MLXTensorSource over a dictionary from safetensors key to array (tests; a fixture file read with
/// MLX.loadArrays).
public struct MLXDictionaryTensorSource: MLXTensorSource {
    /// The arrays by key.
    public let arrays: [String: MLXArray]

    /// Wraps `arrays`.
    public init(arrays: [String: MLXArray]) {
        self.arrays = arrays
    }

    public func tensor(_ key: String) throws -> MLXArray {
        guard let array = arrays[key] else {
            throw ModelPortError.missingTensor(key: key)
        }
        return array
    }
}

extension LoadedWeights: MLXTensorSource {
    /// The checked array of `key` (SafetensorsLoader.swift, array(forKey:)).
    public func tensor(_ key: String) throws -> MLXArray {
        guard let array = array(forKey: key) else {
            throw ModelPortError.missingTensor(key: key)
        }
        return array
    }
}

// MARK: - Element-wise operations and conversions

/// The element-wise operations the layers share, and the conversion of UEMACore arrays to MLX.
public enum NeuralOps {
    /// 1 / sqrt(2), the GELU argument's scale.
    static let inverseSquareRootOfTwo: Float = 0.707_106_781_186_547_5

    /// GELU, exact form: 0.5 x (1 + erf(x / sqrt 2)) (torch.nn.GELU with approximate='none', the
    /// default every encoder, block and head of the released model uses: swin_patch.py:38, 64;
    /// cross_patch.py:52, 70; conv1d_stack.py:221; mlp_project.py:302; forward_model.py:82;
    /// prediction_heads.py:43).
    public static func gelu(_ x: MLXArray) -> MLXArray {
        x * Float(0.5) * (erf(x * inverseSquareRootOfTwo) + Float(1))
    }

    /// x divided by its Euclidean length over the last axis. torch.nn.functional.normalize divides
    /// by max(length, 1e-12) (swin_patch.py:148-149); here the length is sqrt(sum of squares +
    /// 1e-24), which equals it for every length above 1e-12 to float32 precision.
    public static func unitLength(_ x: MLXArray) -> MLXArray {
        x * (x.square().sum(axis: -1, keepDims: true) + Float(1e-24)).rsqrt()
    }

    /// `array` as an MLX array with a leading batch extent of 1: shape [1] + array.shape.
    /// `expected`, when given, is checked first (-1 accepts any extent).
    public static func batched(_ array: ShapedArray, expected: [Int]? = nil, what: String) throws -> MLXArray {
        if let expected {
            let fits = expected.count == array.shape.count
                && zip(expected, array.shape).allSatisfy { pair in pair.0 == -1 || pair.0 == pair.1 }
            guard fits else {
                throw ModelPortError.inputShape(what: what, expected: expected, found: array.shape)
            }
        }
        return MLXArray(array.values, [1] + array.shape)
    }

    /// The numbers of `array` as Swift floats, row-major.
    public static func floats(_ array: MLXArray) -> [Float] {
        array.asType(.float32).asArray(Float.self)
    }

    /// True when every number of `array` is finite (no NaN, no infinity).
    public static func isFinite(_ array: MLXArray) -> Bool {
        floats(array).allSatisfy { $0.isFinite }
    }
}

// MARK: - Linear

/// torch.nn.Linear: y = x W^T + b over the last axis, W [outputs, inputs], b [outputs] or none.
public struct LinearLayer {
    /// The weight [outputs, inputs], PyTorch layout.
    public let weight: MLXArray
    /// The bias [outputs]; nil for a layer built with bias=False.
    public let bias: MLXArray?
    private let weightTransposed: MLXArray

    /// A layer over given arrays.
    public init(weight: MLXArray, bias: MLXArray?) {
        self.weight = weight
        self.bias = bias
        self.weightTransposed = weight.T
    }

    /// The layer stored under `prefix` ("<prefix>weight", and "<prefix>bias" when `hasBias`),
    /// shapes checked against `inputs` and `outputs`.
    public init(source: any MLXTensorSource, prefix: String, inputs: Int, outputs: Int, hasBias: Bool) throws {
        let weight = try source.tensor(prefix + "weight", shape: [outputs, inputs])
        var bias: MLXArray? = nil
        if hasBias {
            bias = try source.tensor(prefix + "bias", shape: [outputs])
        }
        self.init(weight: weight, bias: bias)
    }

    /// The parameter count: outputs x inputs, plus outputs with a bias.
    public var parameterCount: Int { weight.size + (bias?.size ?? 0) }

    /// y = x W^T + b for x [..., inputs].
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = x.matmul(weightTransposed)
        if let bias {
            return y + bias
        }
        return y
    }
}

// MARK: - Revived units of a Linear layer

/// The liveness repair's revived units of one Linear layer (RevivedRows of
/// Source/Python/lem/model/liveness.py:290-361; the released run's --liveness-repair on): for each unit
/// j with mask[j] true, the layer's output (its pre-activation) is replaced by
/// lin(LayerNorm(x - centre))[j], where x is the input the layer itself received; the other units keep
/// the layer's own output (liveness.py:355-361, torch.where(mask, z, pre)). mask [outputs] BOOL, norm
/// LayerNorm(inputs), lin Linear(inputs -> outputs) with bias are stored under
/// "liveness_repair.rows.<layer>." in model.safetensors; centre [inputs] is a running mean registered
/// with persistent=False (liveness.py:309), so it is not in model.safetensors: it is read from
/// Resources/prediction_head_centres.json, which
/// Tests/UEMAModelTests/Fixtures/prediction_heads/export_prediction_head_centres.py exported from
/// runtime_state.pt of checkpoint 494,025.
public struct RevivedRowsLayer {
    /// True for a revived unit, [outputs].
    let mask: MLXArray
    let norm: LayerNormLayer
    let linear: LinearLayer
    /// The running centre of the layer's input, [inputs].
    let centre: MLXArray
    /// Units whose output this layer replaces.
    public let revivedCount: Int

    /// The revived units stored under `prefix` ("liveness_repair.rows.<layer>.") with the exported
    /// `centre` of `inputs` numbers.
    public init(source: any MLXTensorSource, prefix: String, inputs: Int, outputs: Int, centre: [Float]) throws {
        guard centre.count == inputs else {
            throw ModelPortError.tensorShape(key: prefix + "centre_mean", expected: [inputs], found: [centre.count])
        }
        mask = try source.tensor(prefix + "mask", shape: [outputs])
        norm = try LayerNormLayer(source: source, prefix: prefix + "norm.", width: inputs)
        linear = try LinearLayer(source: source, prefix: prefix + "lin.", inputs: inputs, outputs: outputs,
                                 hasBias: true)
        self.centre = MLXArray(centre, [inputs])
        revivedCount = Int(mask.asType(.int32).sum().item(Int32.self))
    }

    /// The parameter count (norm and lin; the mask is a buffer).
    public var parameterCount: Int { norm.parameterCount + linear.parameterCount }

    /// The layer's output `preActivation` [..., outputs] with the revived units replaced, given the
    /// layer's own input `input` [..., inputs].
    public func apply(input: MLXArray, preActivation: MLXArray) -> MLXArray {
        which(mask, linear(norm(input - centre)), preActivation)
    }
}

// MARK: - LayerNorm

/// torch.nn.LayerNorm over the last axis: (x - mean) / sqrt(variance + eps) x weight + bias, with
/// the biased variance (the mean of squared deviations) and eps 1e-5 (the PyTorch default; no
/// released LayerNorm sets another). weight and bias nil for elementwise_affine=False (the
/// registry's shared output norm, encoder_registry.py:78-82).
public struct LayerNormLayer {
    /// The gain [width]; nil without affine parameters.
    public let weight: MLXArray?
    /// The shift [width]; nil without affine parameters.
    public let bias: MLXArray?
    /// The variance floor: 1e-5.
    public let eps: Float

    /// A norm over given arrays (nil, nil: no affine parameters).
    public init(weight: MLXArray?, bias: MLXArray?, eps: Float = 1e-5) {
        self.weight = weight
        self.bias = bias
        self.eps = eps
    }

    /// The norm stored under `prefix` ("<prefix>weight", "<prefix>bias", each [width]).
    public init(source: any MLXTensorSource, prefix: String, width: Int) throws {
        self.init(weight: try source.tensor(prefix + "weight", shape: [width]),
                  bias: try source.tensor(prefix + "bias", shape: [width]))
    }

    /// The parameter count: 2 x width with affine parameters, 0 without.
    public var parameterCount: Int { (weight?.size ?? 0) + (bias?.size ?? 0) }

    /// The normalised x [..., width].
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let centred = x - x.mean(axis: -1, keepDims: true)
        let variance = centred.square().mean(axis: -1, keepDims: true)
        var y = centred * (variance + eps).rsqrt()
        if let weight {
            y = y * weight
        }
        if let bias {
            y = y + bias
        }
        return y
    }
}

// MARK: - Multi-head attention

/// torch.nn.MultiheadAttention with batch_first=True, built from three Linear layers, a softmax
/// and an output Linear: in_proj_weight [3D, D] holds the query, key and value projections in that
/// row sequence (and in_proj_bias [3D] their biases), each head reads D / heads consecutive numbers
/// of the projected vector, scores are q k^T / sqrt(D / heads), softmax runs over the keys, and the
/// heads' outputs are concatenated and passed through out_proj.
public struct MultiheadAttentionLayer {
    /// The model width D.
    public let width: Int
    /// The head count.
    public let heads: Int
    let query: LinearLayer
    let key: LinearLayer
    let value: LinearLayer
    let output: LinearLayer
    private let scale: Float

    /// The attention stored under `prefix` ("<prefix>in_proj_weight", "<prefix>in_proj_bias",
    /// "<prefix>out_proj.weight", "<prefix>out_proj.bias"; the biases only when `hasBias`).
    public init(source: any MLXTensorSource, prefix: String, width: Int, heads: Int, hasBias: Bool) throws {
        precondition(heads > 0 && width % heads == 0, "the width \(width) is a multiple of the head count \(heads)")
        let inWeight = try source.tensor(prefix + "in_proj_weight", shape: [3 * width, width])
        let weights = inWeight.split(parts: 3, axis: 0)
        var biases: [MLXArray?] = [nil, nil, nil]
        if hasBias {
            let inBias = try source.tensor(prefix + "in_proj_bias", shape: [3 * width])
            biases = inBias.split(parts: 3, axis: 0).map { Optional($0) }
        }
        self.width = width
        self.heads = heads
        self.query = LinearLayer(weight: weights[0], bias: biases[0])
        self.key = LinearLayer(weight: weights[1], bias: biases[1])
        self.value = LinearLayer(weight: weights[2], bias: biases[2])
        self.output = try LinearLayer(source: source, prefix: prefix + "out_proj.", inputs: width,
                                      outputs: width, hasBias: hasBias)
        self.scale = 1 / Float(width / heads).squareRoot()
    }

    /// The parameter count: 4 D^2, plus 4 D with biases.
    public var parameterCount: Int {
        query.parameterCount + key.parameterCount + value.parameterCount + output.parameterCount
    }

    /// Attention of `query` [B, L, D] over `key` and `value` [B, S, D]; `additiveMask`, when given,
    /// is added to the scores [B, heads, L, S] by broadcasting (an [L, S] mask of 0 and -infinity).
    /// Returns [B, L, D].
    public func callAsFunction(query queryInput: MLXArray, key keyInput: MLXArray, value valueInput: MLXArray,
                               additiveMask: MLXArray? = nil) -> MLXArray {
        let batch = queryInput.shape[0]
        let queryLength = queryInput.shape[1]
        let keyLength = keyInput.shape[1]
        let headWidth = width / heads
        let q = query(queryInput).reshaped([batch, queryLength, heads, headWidth]).transposed(0, 2, 1, 3)
        let k = key(keyInput).reshaped([batch, keyLength, heads, headWidth]).transposed(0, 2, 1, 3)
        let v = value(valueInput).reshaped([batch, keyLength, heads, headWidth]).transposed(0, 2, 1, 3)
        var scores = q.matmul(k.transposed(0, 1, 3, 2)) * scale
        if let additiveMask {
            scores = scores + additiveMask
        }
        let weights = softmax(scores, axis: -1)
        let mixed = weights.matmul(v).transposed(0, 2, 1, 3).reshaped([batch, queryLength, width])
        return output(mixed)
    }

    /// The additive form of torch.triu(ones(n, n), diagonal=1) as a boolean attn_mask
    /// (forward_model.py:66-69): 0 where key j <= query i, -infinity where j > i. [n, n].
    public static func causalMask(length: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: length * length)
        // row + 1 <= length for every row, so the inner range is valid and empty on the last row
        for row in 0 ..< length {
            for column in (row + 1) ..< length {
                values[row * length + column] = -Float.infinity
            }
        }
        return MLXArray(values, [length, length])
    }
}

// MARK: - The pre-norm attention block of the encoders

/// One pre-norm residual attention block: h = x + attention(norm1(x)); out = h + ffn(norm2(h)) with
/// ffn = Linear(D, 4D), GELU, Linear(4D, D) (Dropout 0.0 at inference). The vision Swin block names
/// its norms norm1 and norm2 (swin_patch.py:57-68, 101-109); the tactile and motor layers name them
/// norm and ffn_norm (cross_patch.py:62-76, 104-112). Both name the feed-forward layers ffn.0 and
/// ffn.3 (Linear, GELU, Dropout, Linear, Dropout).
public struct PreNormAttentionBlock {
    let attentionNorm: LayerNormLayer
    let attention: MultiheadAttentionLayer
    let feedForwardNorm: LayerNormLayer
    let feedForwardIn: LinearLayer
    let feedForwardOut: LinearLayer

    /// The block stored under `prefix`, with its two norms named `attentionNormName` and
    /// `feedForwardNormName`.
    public init(source: any MLXTensorSource, prefix: String, attentionNormName: String, feedForwardNormName: String,
                width: Int, heads: Int, expansion: Int, hasBias: Bool) throws {
        attentionNorm = try LayerNormLayer(source: source, prefix: prefix + attentionNormName + ".", width: width)
        attention = try MultiheadAttentionLayer(source: source, prefix: prefix + "attn.", width: width,
                                                heads: heads, hasBias: hasBias)
        feedForwardNorm = try LayerNormLayer(source: source, prefix: prefix + feedForwardNormName + ".", width: width)
        feedForwardIn = try LinearLayer(source: source, prefix: prefix + "ffn.0.", inputs: width,
                                        outputs: width * expansion, hasBias: hasBias)
        feedForwardOut = try LinearLayer(source: source, prefix: prefix + "ffn.3.", inputs: width * expansion,
                                         outputs: width, hasBias: hasBias)
    }

    /// The parameter count of the block.
    public var parameterCount: Int {
        attentionNorm.parameterCount + attention.parameterCount + feedForwardNorm.parameterCount
            + feedForwardIn.parameterCount + feedForwardOut.parameterCount
    }

    /// The block on x [B, N, D]; self-attention among the N elements.
    public func callAsFunction(_ x: MLXArray, additiveMask: MLXArray? = nil) -> MLXArray {
        let normed = attentionNorm(x)
        let attended = attention(query: normed, key: normed, value: normed, additiveMask: additiveMask) + x
        return feedForwardOut(NeuralOps.gelu(feedForwardIn(feedForwardNorm(attended)))) + attended
    }
}

// MARK: - The forward model's three-stage block

/// One block of the forward model (forward_model.py:34-184): cross-modal attention per model step
/// over its 6 sense tokens, causal-temporal attention per sense over the 128 model steps, the
/// bridge over all 768 tokens, then the feed-forward layer; each stage pre-norm with a residual.
public struct ThreeStageBlock {
    let senseCount: Int
    let contextLength: Int
    let width: Int
    let crossModalNorm: LayerNormLayer
    let crossModal: MultiheadAttentionLayer
    let temporalNorm: LayerNormLayer
    let temporal: MultiheadAttentionLayer
    let bridgeNorm: LayerNormLayer
    let bridge: MultiheadAttentionLayer
    let feedForwardNorm: LayerNormLayer
    let feedForwardIn: LinearLayer
    let feedForwardOut: LinearLayer
    let causalMask: MLXArray

    /// The block stored under `prefix` ("forward_model.blocks.<n>." in the released file).
    public init(source: any MLXTensorSource, prefix: String, width: Int, heads: Int, expansion: Int,
                senseCount: Int, contextLength: Int) throws {
        self.senseCount = senseCount
        self.contextLength = contextLength
        self.width = width
        crossModalNorm = try LayerNormLayer(source: source, prefix: prefix + "cm_norm.", width: width)
        crossModal = try MultiheadAttentionLayer(source: source, prefix: prefix + "cm_attn.", width: width,
                                                 heads: heads, hasBias: true)
        temporalNorm = try LayerNormLayer(source: source, prefix: prefix + "ct_norm.", width: width)
        temporal = try MultiheadAttentionLayer(source: source, prefix: prefix + "ct_attn.", width: width,
                                               heads: heads, hasBias: true)
        bridgeNorm = try LayerNormLayer(source: source, prefix: prefix + "cmb_norm.", width: width)
        bridge = try MultiheadAttentionLayer(source: source, prefix: prefix + "cmb_attn.", width: width,
                                             heads: heads, hasBias: true)
        feedForwardNorm = try LayerNormLayer(source: source, prefix: prefix + "ffn_norm.", width: width)
        feedForwardIn = try LinearLayer(source: source, prefix: prefix + "ffn.0.", inputs: width,
                                        outputs: width * expansion, hasBias: true)
        feedForwardOut = try LinearLayer(source: source, prefix: prefix + "ffn.3.", inputs: width * expansion,
                                         outputs: width, hasBias: true)
        causalMask = MultiheadAttentionLayer.causalMask(length: contextLength)
    }

    /// The parameter count of the block: 742,080 at width 192, expansion 4.
    public var parameterCount: Int {
        crossModalNorm.parameterCount + crossModal.parameterCount + temporalNorm.parameterCount
            + temporal.parameterCount + bridgeNorm.parameterCount + bridge.parameterCount
            + feedForwardNorm.parameterCount + feedForwardIn.parameterCount + feedForwardOut.parameterCount
    }

    /// The block on the sequence x [B, contextLength x senseCount, D], laid out model step by model
    /// step with the senses interleaved inside each model step.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.shape[0]
        let sequence = contextLength * senseCount
        // Stage 1: per model step, across its 6 sense tokens (forward_model.py:124-140).
        let perStep = x.reshaped([batch * contextLength, senseCount, width])
        let perStepNormed = crossModalNorm(perStep)
        let afterCrossModal = (crossModal(query: perStepNormed, key: perStepNormed, value: perStepNormed) + perStep)
            .reshaped([batch, sequence, width])
        // Stage 2: per sense, across the 128 model steps, causal (forward_model.py:143-161).
        let perSense = afterCrossModal.reshaped([batch, contextLength, senseCount, width])
            .transposed(0, 2, 1, 3)
            .reshaped([batch * senseCount, contextLength, width])
        let perSenseNormed = temporalNorm(perSense)
        let afterTemporal = (temporal(query: perSenseNormed, key: perSenseNormed, value: perSenseNormed,
                                      additiveMask: causalMask) + perSense)
            .reshaped([batch, senseCount, contextLength, width])
            .transposed(0, 2, 1, 3)
            .reshaped([batch, sequence, width])
        // Stage 3: the bridge over all 768 tokens, no mask (forward_model.py:164-172).
        let bridgeNormed = bridgeNorm(afterTemporal)
        let afterBridge = bridge(query: bridgeNormed, key: bridgeNormed, value: bridgeNormed) + afterTemporal
        // Feed-forward per token (forward_model.py:175-177).
        return feedForwardOut(NeuralOps.gelu(feedForwardIn(feedForwardNorm(afterBridge)))) + afterBridge
    }
}

// MARK: - The transformer

/// What one pass of the transformer gives (forward_model.py:318-325): latents [B, 768, 192] (the
/// whole sequence after final_norm and the memory bias, read by the motor controller and wellness,
/// WI-36), lastStep [B, 6, 192] (the last model step's 6 tokens, read by the prediction heads) and
/// contextUpdate [B, 6, 192] (this model step's tokens plus the modality embedding, before the
/// blocks: the frame the context buffer appends).
public struct TransformerOutput {
    /// The whole sequence, [B, 768, 192].
    public let latents: MLXArray
    /// The last model step's 6 tokens, [B, 6, 192].
    public let lastStep: MLXArray
    /// This model step's frame for the context buffer, [B, 6, 192].
    public let contextUpdate: MLXArray
}

/// The forward model's transformer (forward_model.py:192-426, inference path): the modality and
/// temporal embeddings, 4 three-stage blocks of width 192 with 8 heads and feed-forward expansion 4,
/// final_norm, and the memory bias on the last model step's 6 tokens.
public struct TransformerCore {
    /// The released sizes (model_receipt.json forward_model; args.json d_model 192, context_length 128).
    public enum Released {
        /// Token width: 192.
        public static let width = 192
        /// Attention heads per stage: 8.
        public static let heads = 8
        /// Three-stage blocks: 4.
        public static let blocks = 4
        /// Feed-forward width per token width: 4.
        public static let feedForwardExpansion = 4
        /// The context in model steps: 128 (ModelStep.contextLength).
        public static let contextLength = ModelStep.contextLength
        /// Tokens per model step: the 6 active senses.
        public static let senseCount = SenseName.activeCount
        /// Tokens in the sequence: 768.
        public static let sequenceLength = contextLength * senseCount
        /// Parameters of the component forward_model: 2,994,432 (model_receipt.json component_params).
        public static let parameterCount = 2_994_432
    }

    /// The modality embedding [6, 192]: row i is added to the i-th active sense's token.
    public let modalityEmbedding: MLXArray
    /// The temporal embedding [128, 192]: row t is added to the 6 tokens of model step t.
    public let temporalEmbedding: MLXArray
    let blocks: [ThreeStageBlock]
    let finalNorm: LayerNormLayer
    let width: Int
    let senseCount: Int
    let contextLength: Int
    private let temporalPerToken: MLXArray

    /// The transformer stored under `source` (the released file's keys seen through
    /// `scoped("forward_model.")`).
    public init(source: any MLXTensorSource, width: Int = Released.width, heads: Int = Released.heads,
                blockCount: Int = Released.blocks, expansion: Int = Released.feedForwardExpansion,
                senseCount: Int = Released.senseCount, contextLength: Int = Released.contextLength) throws {
        self.width = width
        self.senseCount = senseCount
        self.contextLength = contextLength
        modalityEmbedding = try source.tensor("modality_embedding.weight", shape: [senseCount, width])
        temporalEmbedding = try source.tensor("temporal_embedding.weight", shape: [contextLength, width])
        blocks = try (0 ..< blockCount).map { index in
            try ThreeStageBlock(source: source, prefix: "blocks.\(index).", width: width, heads: heads,
                                expansion: expansion, senseCount: senseCount, contextLength: contextLength)
        }
        finalNorm = try LayerNormLayer(source: source, prefix: "final_norm.", width: width)
        // forward_model.py:287-291: row t repeated for the 6 tokens of model step t, [768, 192].
        temporalPerToken = broadcast(temporalEmbedding.expandedDimensions(axis: 1),
                                     to: [contextLength, senseCount, width])
            .reshaped([contextLength * senseCount, width])
    }

    /// The parameter count: 2,994,432 for the released sizes.
    public var parameterCount: Int {
        modalityEmbedding.size + temporalEmbedding.size + finalNorm.parameterCount
            + blocks.reduce(0) { $0 + $1.parameterCount }
    }

    /// The tokens of this model step [B, 6, 192] plus the modality embedding: the frame the
    /// context buffer keeps (forward_model.py:367-372), also used for the motor slot back-fill.
    public func embedded(_ tokens: MLXArray) -> MLXArray {
        tokens + modalityEmbedding
    }

    /// One pass. `tokens` [B, 6, 192] are this model step's tokens after the shared output
    /// LayerNorm, in the active sense sequence; `context` [B, 762, 192] the previous 127 model
    /// steps' frames (zeros after a reset); `memoryBias` [B, 6, 192] or nil.
    public func callAsFunction(tokens: MLXArray, context: MLXArray, memoryBias: MLXArray?) throws -> TransformerOutput {
        let batch = tokens.shape[0]
        guard tokens.shape == [batch, senseCount, width] else {
            throw ModelPortError.inputShape(what: "transformer tokens", expected: [batch, senseCount, width],
                                            found: tokens.shape)
        }
        let contextTokens = (contextLength - 1) * senseCount
        guard context.shape == [batch, contextTokens, width] else {
            throw ModelPortError.inputShape(what: "transformer context", expected: [batch, contextTokens, width],
                                            found: context.shape)
        }
        let current = embedded(tokens)
        var sequence = concatenated([context, current], axis: 1) + temporalPerToken
        for block in blocks {
            sequence = block(sequence)
        }
        sequence = finalNorm(sequence)
        if let memoryBias {
            guard memoryBias.shape == [batch, senseCount, width] else {
                throw ModelPortError.inputShape(what: "memory bias", expected: [batch, senseCount, width],
                                                found: memoryBias.shape)
            }
            // forward_model.py:415-419: the bias reaches the last model step's 6 tokens only.
            sequence = concatenated([sequence[0..., 0 ..< contextTokens],
                                     sequence[0..., contextTokens...] + memoryBias], axis: 1)
        }
        let lastStep = sequence[0..., contextTokens...]
        return TransformerOutput(latents: sequence, lastStep: lastStep, contextUpdate: current)
    }
}
