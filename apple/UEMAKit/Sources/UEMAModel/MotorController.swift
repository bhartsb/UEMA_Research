// MotorController.swift: the released motor controller of the UEMAModel module (work item WI-36
// of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-10 of
// docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html; contract apple/UEMAKit/Specs/
// motor_controller_input.md). Nothing in this file was compiled on the machine that wrote it
// (Windows 11, no Swift toolchain, no Xcode: plan section 3, fact 1).
//
// WHAT THIS FILE DOES.
// INPUT, once per model step (one 50 ms model update): 1,646 numbers (motor_controller_input.md
//   section 1; receipt motor_controller._input_dim 1646):
//   - the forward model's newest output, 6 senses x 192 = 1,152 numbers, in the sequence vision,
//     tactile, audio, proprioception, motor, vestibular (motor_controller.py:146-147,
//     `latents[:, -6:, :]`; the fixtures' forward/last_step [1, 6, 192]);
//   - the handle state, 26 body handles x 10 numbers = 260 (the achieved motor half: position
//     cm / 200, quaternion x y z w, commanded radius, commanded strength, mask 1.0;
//     proprio_rows.md section 5; motor_controller.py:159);
//   - the wellness gradient, 26 x 9 = 234 numbers: d(wellness sum) / d(centred handle pose),
//     computed by Wellness.swift's MotorStateBoundary.gradient (lem_model.py:2999-3001,
//     :3099-3106; motor_controller_input.md section 2).
// OPERATION (motor_controller.py:121-229 with the released settings adaptive true, max_depth 4,
//   hidden 128; plus the two inference-time hooks described below):
//   1. read = newest latents - c, where c = persistent_memory.readout.bias (1,152 numbers) while
//      motor_input_fold.folded is 1 (motor_input_fold.py:183-198, the pre-hooks on the input
//      projection's 1,152 token columns and on the confidence head; effective_constant at
//      :95-104 with the constant's return off: persistent_memory.constant_return_updates 0 in
//      the released file, persistent_memory.py:394-407 then gives a factor of exactly 1).
//   2. confidence = sigmoid(L2(gelu(L0'(read)))), L0 = confidence_head.0 Linear(1152, 128),
//      L2 = confidence_head.2 Linear(128, 1) (motor_controller.py:66-71, :150). L0' is L0 with
//      its revived units replaced (liveness.py:355-361): for each unit j with mask[j] 1, the
//      pre-activation is lin(LayerNorm(read - centre))[j], lin = liveness_repair.rows.
//      motor_controller__confidence_head__0.lin Linear(1152, 128), LayerNorm over 1,152 numbers
//      with eps 1e-5 (torch.nn.LayerNorm's default) and the rows' norm weight and bias; centre =
//      the rows' running centre_mean (1,152 numbers, runtime_state.pt only, exported to
//      Resources/motor_controller_centres.json). 84 of the 128 units are revived (receipt
//      liveness_repair.revived).
//   3. x = [read (1,152); handle state (260); wellness gradient (234)] = 1,646 numbers
//      (motor_controller.py:160).
//   4. depth threshold t = sigmoid(depth_threshold_raw); raw depth d = 1 + 3 sigmoid(5 (t -
//      confidence)) (motor_controller.py:163-168).
//   5. h = input_proj(x), Linear(1646, 128); for block i = 0 ... 3: w_i = sigmoid(5 (d - (i +
//      1))); h = w_i * B_i(h) + h, B_i = Linear(128, 128), GELU, Linear(128, 128)
//      (motor_controller.py:183-186: all four blocks run, the soft gate is the released
//      behaviour at inference). Block 0's first Linear has 127 of 128 units revived, the same
//      rule as step 2 with its input h (128 numbers) and its own centre (128 numbers).
//   6. o = output_proj(h), Linear(128, 234), as 26 rows of 9 (motor_controller.py:188, :204).
//   7. Per row (motor_controller.py:224-229): position = clamp(o[0..<3], -1, 1); rotation =
//      o[3..<7] / max(|o[3..<7]|, 1e-8); radius, strength = sigmoid(o[7]), sigmoid(o[8]).
// OUTPUT: the command, 26 body handles x 9 numbers = 234, positions in position_range units
//   (scale `.normalized`; x 200 = cm after the execution transform, planner_loss.md section 3
//   step 2), with the confidence (0 to 1), the raw depth (1 to 4), the depth threshold and the
//   78 position excursions max(|o| - 1, 0) (motor_controller.py:236).
//
// WHAT THE RELEASED CONTROLLER COMPUTES: MEASURED, NOT ASSUMED. motor_controller_input.md left
// two points open (section 6, A: the runtime effect of motor_input_fold; B: of liveness_repair).
// Both were measured on November 4, 2026 against the three model-pass fixtures of WI-34
// (Tests/UEMAModelTests/Fixtures/model_pass/step_000492800, _000493674, _000493980; tensors
// forward/latents, command/handle_state, command/wellness_grad, command/motor_out) with
// model.safetensors and runtime_state.pt of ckpt_step_494025, by a torch reproduction of the
// steps above. Largest absolute difference from command/motor_out over the 234 numbers, worst
// of the three states:
//   steps 1 to 7 as written, float64:                          5.71e-6
//   steps 1 to 7 as written, float32:                          2.38e-6
//   without the revived units (plain L0 and block 0):          1.39
//   without the fold (read = newest latents):                  1.47
// on a command whose numbers lie between -1 and 1. So both hooks are part of the released
// controller, and both are implemented here. All three states have confidence 1.0 and raw
// depth 2.0794 (depth threshold 0.8848), so these fixtures exercise block weights 0.5, 0.0003,
// 1.4e-7 and 7e-10 for blocks 0 to 3 only.
//
// WHERE THE COMPUTATION RUNS. The ten Linear layers multiply on MLX (addMM on the default
// device, Metal on Apple silicon). The element-wise steps (GELU, sigmoid, LayerNorm of the 84 +
// 127 revived units' input, the per-row activations) run on the CPU in Swift, in Double, on at
// most 1,646 numbers. The AffineLayerBuilding protocol picks the layer implementation:
// MLXAffineLayerBuilder for the app, ReferenceAffineLayerBuilder (plain Swift loops) for tests
// that run without a Metal device and for cross-checking MLX on the Mac. This controller is not
// differentiated: the planner's gradient goes through the forward model only (REQ-B-17; plan row
// WI-39), so no MLX graph is kept across the layers.
//
// MLX SWIFT SYMBOLS USED, each read on November 4, 2026 from the source at
// https://raw.githubusercontent.com/ml-explore/mlx-swift/main/ (the documentation pages at
// https://swiftpackageindex.com/ml-explore/mlx-swift/documentation/mlx carry the same
// declarations):
// - MLXArray.T (the transpose of every axis; for the [out, in] weight matrix it is the [in, out]
//   matrix) and MLXArray.matmul(_:), the two members TransformerCore.swift's Linear also uses
//   (`weight.T`, `x.matmul(weightTransposed)`), and `+` with broadcasting of the [out] bias.
//   Source/MLX/Ops.swift on main also declares addMM and matrixTransposed; this file does not
//   use them, because matrixTransposed calls mlx_matrix_transpose of mlx-c, which an older
//   resolved tag (0.31.4, the tag a Swift 6.0 to 6.2 toolchain resolves, Package.swift header)
//   may not carry.
// - Source/MLX/MLXArray+Init.swift: "public convenience init<T: HasDType>(_ value: [T], _ shape:
//   (some Collection<Int>)? = [Int]?.none)", example "MLXArray([0, 1, 2, 3], [2, 2])".
// - Source/MLX/MLXArray.swift: "final public func asType(_ type: DType, stream: StreamOrDevice =
//   .default) -> MLXArray".
// - MLXArray.asArray(_:) ("asArray<T: HasDType>(_ type: T.Type) -> [T]"): not in the four
//   source files read; see FIRST BUILD ON A MAC, step 3.
// - Source/MLXNN/Activations.swift: gelu is "x * (1 + MLX.erf(x / sqrt(2))) / 2", the exact
//   form, which equals torch.nn.GELU's default (approximate 'none'); this file computes the
//   same formula on the CPU with Foundation's erf.
// Foundation: erf(Double), exp(Double), tanh(Double), log10(Double)
// (https://developer.apple.com/documentation/foundation, the C math functions of Darwin).
//
// SOLID. S: DenseWeights holds one Linear layer's numbers; AffineLayer applies one; RevivedRows
// applies the liveness routing of one layer; RevivedCentres decodes the exported centres;
// MotorControllerParameters reads and checks every tensor the controller needs; MotorController
// runs steps 1 to 7. D: MotorController depends on the AffineLayer and AffineLayerBuilding
// protocols and reads tensors through TensorSource (LoadedWeights conforms below; the tests
// inject a dictionary).
//
// FIRST BUILD ON A MAC (this work item's acceptance: "shapes match the manifest; the fixture's
// command and wellness score matched within the recorded tolerance").
//  1. apple/README.md "The Mac first-build sequence" steps 1 to 3 have passed (weights fetched
//     to weights/b1_continue_B1K_skip13_492800/run/checkpoints/ckpt_step_494025/model.safetensors,
//     MuJoCo XCFramework built, `swift package resolve` exit 0).
//  2. `cd apple/UEMAKit && xcodebuild -scheme UEMAKit test -destination
//     'platform=macOS' -only-testing:UEMAModelTests/MotorControllerTests
//     -only-testing:UEMAModelTests/WellnessTests` : exit code 0, 26 tests (MotorControllerTests
//     14, WellnessTests 12). Without the weights file
//     MotorControllerTests.testReleasedControllerReproducesTheFixtures skips and the rest run
//     (manifest shapes, hand-set networks, MLX against the plain-Swift layer, the closed-form
//     wellness gradient against torch's numbers, tactile comfort against contact_wellness.py's
//     numbers, the fixture wellness sum, which reads the committed fixtures only).
//  3. If `asArray(Float.self)` does not resolve, read Source/MLX/MLXArray+Bytes.swift (or the
//     file that declares it) of the resolved mlx-swift tag and replace the two calls in this
//     file (LoadedWeights.tensor(named:) and MLXAffineLayer.apply) with its current form; the
//     same call is in TransformerCore.swift and in ControllerFixture of MotorControllerTests.swift.
//     If `.T` or `matmul(_:)` does not resolve in MLXAffineLayer, use the form TransformerCore.swift's
//     Linear compiles with on that tag.
//  4. The real-file tests print one line per fixture state: "motor controller <state>: largest
//     |difference| <x> over 234 numbers (MLX), <y> (reference)". The pass line is 1e-4 (the
//     float32 torch reproduction measured 2.38e-6 at most, so the line leaves a margin of 42
//     times that for MLX's Metal summation sequence). Write the printed numbers and the command,
//     exit code and clock time (AM/PM Pacific) into apple/FIRST_BUILD_RECORD.md (WI-41) and the
//     plan's section 9.
//  5. If the real-file test fails with a difference above 1e-4 on every state, check first that
//     Resources/motor_controller_centres.json was exported from the same runtime_state.pt (its
//     source.runtime_state_sha256, 46b40ef7...5347) and that motor_input_fold.folded reads 1.

import Foundation
import MLX
import MLXNN
import UEMACore

// MARK: - Errors

/// Every way reading the controller's tensors or running it refuses; each case names the tensor
/// or input involved.
public enum MotorControllerError: Error, Equatable, CustomStringConvertible {
    /// The tensor source holds no tensor of this key.
    case missingTensor(String)
    /// The tensor of `key` has shape `found`; the controller needs `expected`.
    case wrongShape(key: String, expected: [Int], found: [Int])
    /// An input array has `found` numbers where `expected` are required.
    case wrongInputCount(name: String, expected: Int, found: Int)
    /// An input array holds a NaN or an infinity.
    case nonFiniteInput(String)
    /// The persistent memory's constant return is on (`repair_switched` 1 and
    /// `constant_return_updates` > 0), so the fold constant is the readout bias times a factor
    /// that this port does not compute (persistent_memory.py:394-407, :463).
    case unsupportedConstantReturn(updates: Float)
    /// Units of `layer` are revived but no centre was given for it.
    case missingCentre(layer: String)
    /// The centres file has another schema.
    case centresSchema(found: String, expected: String)
    /// The centres resource is not in the bundle.
    case centresNotFound(resource: String)

    public var description: String {
        switch self {
        case .missingTensor(let key):
            return "the tensor source holds no tensor \(key)"
        case .wrongShape(let key, let expected, let found):
            return "\(key) has shape \(found); the motor controller needs \(expected)"
        case .wrongInputCount(let name, let expected, let found):
            return "\(name) has \(found) numbers; the motor controller needs \(expected)"
        case .nonFiniteInput(let name):
            return "\(name) holds a NaN or an infinity"
        case .unsupportedConstantReturn(let updates):
            return "persistent_memory.constant_return_updates is \(updates); this port computes the fold constant only for 0"
        case .missingCentre(let layer):
            return "\(layer) has revived units but no running centre was given"
        case .centresSchema(let found, let expected):
            return "motor controller centres schema is \(found); this reader reads \(expected)"
        case .centresNotFound(let resource):
            return "motor controller centres resource \(resource) is not in the bundle"
        }
    }
}

// MARK: - Tensors by key

/// Gives one saved tensor by its safetensors key as float32 numbers with its shape. The
/// controller reads its tensors through this protocol, so a test can hand it a dictionary.
public protocol TensorSource {
    /// True when a tensor of `key` exists.
    func hasTensor(named key: String) -> Bool

    /// The tensor of `key` converted to float32 (a BOOL tensor gives 0 and 1); throws
    /// `MotorControllerError.missingTensor` when absent.
    func tensor(named key: String) throws -> ShapedArray
}

extension LoadedWeights: TensorSource {
    public func hasTensor(named key: String) -> Bool {
        self.array(forKey: key) != nil
    }

    public func tensor(named key: String) throws -> ShapedArray {
        guard let stored = self.array(forKey: key) else {
            throw MotorControllerError.missingTensor(key)
        }
        let values = stored.asType(.float32).asArray(Float.self)
        return try ShapedArray(shape: stored.shape, values: values)
    }
}

/// A TensorSource over a dictionary of arrays already in memory (tests, and fixtures read by
/// other means).
public struct DictionaryTensorSource: TensorSource {
    /// The tensors by safetensors key.
    public let tensors: [String: ShapedArray]

    public init(tensors: [String: ShapedArray]) {
        self.tensors = tensors
    }

    public func hasTensor(named key: String) -> Bool {
        tensors[key] != nil
    }

    public func tensor(named key: String) throws -> ShapedArray {
        guard let tensor = tensors[key] else {
            throw MotorControllerError.missingTensor(key)
        }
        return tensor
    }
}

/// Reads `key` and checks its shape.
func readTensor(_ key: String, shape: [Int], from source: any TensorSource) throws -> ShapedArray {
    let tensor = try source.tensor(named: key)
    guard tensor.shape == shape else {
        throw MotorControllerError.wrongShape(key: key, expected: shape, found: tensor.shape)
    }
    return tensor
}

// MARK: - One Linear layer

/// The numbers of one torch.nn.Linear layer: `weight` [outputCount, inputCount] row-major (the
/// PyTorch layout, as model.safetensors stores it) and `bias` [outputCount]. y = W x + b.
public struct DenseWeights: Equatable, Sendable {
    public let inputCount: Int
    public let outputCount: Int
    /// outputCount x inputCount numbers, row j = the weights of output unit j.
    public let weight: [Float]
    /// outputCount numbers.
    public let bias: [Float]

    /// Checks that `weight` has outputCount x inputCount numbers and `bias` outputCount.
    public init(inputCount: Int, outputCount: Int, weight: [Float], bias: [Float]) throws {
        guard weight.count == inputCount * outputCount else {
            throw MotorControllerError.wrongInputCount(name: "weight", expected: inputCount * outputCount, found: weight.count)
        }
        guard bias.count == outputCount else {
            throw MotorControllerError.wrongInputCount(name: "bias", expected: outputCount, found: bias.count)
        }
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.weight = weight
        self.bias = bias
    }

    /// Reads `<prefix>.weight` [outputCount, inputCount] and `<prefix>.bias` [outputCount].
    public static func read(prefix: String, inputCount: Int, outputCount: Int,
                            from source: any TensorSource) throws -> DenseWeights {
        let weight = try readTensor(prefix + ".weight", shape: [outputCount, inputCount], from: source)
        let bias = try readTensor(prefix + ".bias", shape: [outputCount], from: source)
        return try DenseWeights(inputCount: inputCount, outputCount: outputCount,
                                weight: weight.values, bias: bias.values)
    }
}

/// Applies one Linear layer to one input vector: y = W x + b.
public protocol AffineLayer {
    var inputCount: Int { get }
    var outputCount: Int { get }

    /// `input` has inputCount numbers (the caller checks); the result has outputCount.
    func apply(_ input: [Float]) -> [Float]
}

/// Makes the AffineLayer of one DenseWeights; injected into MotorController.
public protocol AffineLayerBuilding {
    func makeLayer(_ weights: DenseWeights) -> any AffineLayer
}

/// The Linear layer on MLX: x [1, in] matmul W^T [in, out] + bias [out] on the default device.
/// It uses MLXArray.T and MLXArray.matmul(_:), the same two members TransformerCore.swift's
/// Linear uses, so that the motor controller and the transformer depend on one set of MLX
/// symbols across the mlx-swift tags Package.swift can resolve (0.31.4 and 0.33.0).
public struct MLXAffineLayer: AffineLayer {
    public let inputCount: Int
    public let outputCount: Int
    private let transposedWeight: MLXArray
    private let bias: MLXArray

    public init(_ weights: DenseWeights) {
        inputCount = weights.inputCount
        outputCount = weights.outputCount
        transposedWeight = MLXArray(weights.weight, [weights.outputCount, weights.inputCount]).T
        bias = MLXArray(weights.bias, [weights.outputCount])
    }

    public func apply(_ input: [Float]) -> [Float] {
        let x = MLXArray(input, [1, inputCount])
        let y = x.matmul(transposedWeight) + bias
        return y.asType(.float32).asArray(Float.self)
    }
}

/// Builds MLXAffineLayer: the app's choice.
public struct MLXAffineLayerBuilder: AffineLayerBuilding {
    public init() {}

    public func makeLayer(_ weights: DenseWeights) -> any AffineLayer {
        MLXAffineLayer(weights)
    }
}

/// The Linear layer in plain Swift, each output summed in Double over the inputs in index
/// sequence, then rounded to Float.
public struct ReferenceAffineLayer: AffineLayer {
    public let weights: DenseWeights
    public var inputCount: Int { weights.inputCount }
    public var outputCount: Int { weights.outputCount }

    public init(_ weights: DenseWeights) {
        self.weights = weights
    }

    public func apply(_ input: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: weights.outputCount)
        for row in 0 ..< weights.outputCount {
            var sum = Double(weights.bias[row])
            let base = row * weights.inputCount
            for column in 0 ..< weights.inputCount {
                sum += Double(weights.weight[base + column]) * Double(input[column])
            }
            out[row] = Float(sum)
        }
        return out
    }
}

/// Builds ReferenceAffineLayer: tests and the cross-check of MLX.
public struct ReferenceAffineLayerBuilder: AffineLayerBuilding {
    public init() {}

    public func makeLayer(_ weights: DenseWeights) -> any AffineLayer {
        ReferenceAffineLayer(weights)
    }
}

// MARK: - Element-wise functions (CPU, Double)

/// The element-wise functions of the controller, in Double: torch.nn.GELU's exact form, the
/// logistic sigmoid, and LayerNorm over one vector.
public enum MotorControllerMath {
    /// gelu(x) = x (1 + erf(x / sqrt 2)) / 2 (torch.nn.GELU, approximate 'none').
    public static func gelu(_ x: Double) -> Double {
        0.5 * x * (1.0 + erf(x / 2.0.squareRoot()))
    }

    /// 1 / (1 + exp(-x)).
    public static func sigmoid(_ x: Double) -> Double {
        1.0 / (1.0 + exp(-x))
    }

    /// LayerNorm over the whole vector: (x - mean) / sqrt(variance + eps) * weight + bias, the
    /// variance the mean of the squared deviations (torch.nn.LayerNorm; eps 1e-5, its default).
    public static func layerNorm(_ x: [Double], weight: [Float], bias: [Float], epsilon: Double = 1e-5) -> [Float] {
        let n = Double(x.count)
        let mean = x.reduce(0, +) / n
        var variance = 0.0
        for value in x {
            variance += (value - mean) * (value - mean)
        }
        variance /= n
        let scale = 1.0 / (variance + epsilon).squareRoot()
        var out = [Float](repeating: 0, count: x.count)
        for index in 0 ..< x.count {
            out[index] = Float((x[index] - mean) * scale * Double(weight[index]) + Double(bias[index]))
        }
        return out
    }
}

// MARK: - Revived units of one layer (liveness_repair)

/// The saved numbers of one layer's revived units (liveness.py:290-361, class RevivedRows):
/// mask [outputCount] (1 = revived), the LayerNorm weight and bias [inputCount], the
/// replacement Linear layer, and the running centre [inputCount] the input is centred on.
public struct RevivedRowsWeights: Equatable, Sendable {
    /// The rows' key in model.safetensors and runtime_state.pt, e.g.
    /// "motor_controller__confidence_head__0".
    public let key: String
    public let mask: [Bool]
    public let normWeight: [Float]
    public let normBias: [Float]
    public let linear: DenseWeights
    public let centre: [Float]

    /// The number of revived units (mask entries that are 1).
    public var revivedCount: Int { mask.filter { $0 }.count }

    public init(key: String, mask: [Bool], normWeight: [Float], normBias: [Float],
                linear: DenseWeights, centre: [Float]) throws {
        let n = linear.inputCount
        guard mask.count == linear.outputCount else {
            throw MotorControllerError.wrongInputCount(name: key + ".mask", expected: linear.outputCount, found: mask.count)
        }
        guard normWeight.count == n, normBias.count == n else {
            throw MotorControllerError.wrongInputCount(name: key + ".norm", expected: n, found: normWeight.count)
        }
        guard centre.count == n else {
            throw MotorControllerError.wrongInputCount(name: key + ".centre_mean", expected: n, found: centre.count)
        }
        self.key = key
        self.mask = mask
        self.normWeight = normWeight
        self.normBias = normBias
        self.linear = linear
        self.centre = centre
    }

    /// Reads `liveness_repair.rows.<key>.{mask, norm.weight, norm.bias, lin.weight, lin.bias}`
    /// and takes the centre from `centre`; nil when the source has no such rows or no unit is
    /// revived (liveness.py:357: "unchanged when none is").
    public static func read(key: String, inputCount: Int, outputCount: Int, centre: [Float]?,
                            from source: any TensorSource) throws -> RevivedRowsWeights? {
        let prefix = "liveness_repair.rows." + key
        guard source.hasTensor(named: prefix + ".mask") else {
            return nil
        }
        let mask = try readTensor(prefix + ".mask", shape: [outputCount], from: source).values.map { $0 != 0 }
        guard mask.contains(true) else {
            return nil
        }
        guard let centre = centre else {
            throw MotorControllerError.missingCentre(layer: key)
        }
        let normWeight = try readTensor(prefix + ".norm.weight", shape: [inputCount], from: source).values
        let normBias = try readTensor(prefix + ".norm.bias", shape: [inputCount], from: source).values
        let linear = try DenseWeights.read(prefix: prefix + ".lin", inputCount: inputCount,
                                           outputCount: outputCount, from: source)
        return try RevivedRowsWeights(key: key, mask: mask, normWeight: normWeight, normBias: normBias,
                                      linear: linear, centre: centre)
    }
}

/// Applies one layer's revived-unit routing: for each revived unit j, the pre-activation is
/// lin(LayerNorm(x - centre))[j]; every other unit keeps the layer's own value
/// (liveness.py:355-361, `torch.where(self.mask, z, pre)`).
public struct RevivedRows {
    public let weights: RevivedRowsWeights
    private let linear: any AffineLayer

    public init(weights: RevivedRowsWeights, builder: any AffineLayerBuilding) {
        self.weights = weights
        self.linear = builder.makeLayer(weights.linear)
    }

    /// `input` is the layer's input (inputCount numbers), `preActivation` the layer's own output
    /// (outputCount numbers).
    public func apply(input: [Float], preActivation: [Float]) -> [Float] {
        var centred = [Double](repeating: 0, count: input.count)
        for index in 0 ..< input.count {
            centred[index] = Double(input[index]) - Double(weights.centre[index])
        }
        let normed = MotorControllerMath.layerNorm(centred, weight: weights.normWeight, bias: weights.normBias)
        let replaced = linear.apply(normed)
        var out = preActivation
        for unit in 0 ..< out.count where weights.mask[unit] {
            out[unit] = replaced[unit]
        }
        return out
    }
}

/// The running centres of the two revived layers of the motor controller, exported from
/// runtime_state.pt (liveness.py:1132-1134) by
/// Tests/UEMAModelTests/Fixtures/motor_controller/export_motor_controller_centres.py into
/// Resources/motor_controller_centres.json (schema uema_motor_controller_centres_v1).
public struct RevivedCentres: Decodable, Equatable, Sendable {
    /// The schema this reader reads.
    public static let expectedSchema = "uema_motor_controller_centres_v1"
    public static let resourceName = "motor_controller_centres"
    public static let resourceExtension = "json"
    public static let resourceSubdirectory = "Resources"

    /// One layer's entry of the file.
    public struct Entry: Decodable, Equatable, Sendable {
        public let key: String
        public let centreWeight: Double
        public let centreMean: [Float]

        enum CodingKeys: String, CodingKey {
            case key
            case centreWeight = "centre_weight"
            case centreMean = "centre_mean"
        }
    }

    public let schema: String
    /// motor_controller__confidence_head__0: 1,152 numbers.
    public let confidenceHead0: Entry
    /// motor_controller__residual_blocks__0__0: 128 numbers.
    public let residualBlock0: Entry

    enum CodingKeys: String, CodingKey {
        case schema
        case confidenceHead0 = "confidence_head_0"
        case residualBlock0 = "residual_blocks_0_0"
    }

    /// Decodes the file and checks the schema.
    public static func decode(from data: Data) throws -> RevivedCentres {
        let centres = try JSONDecoder().decode(RevivedCentres.self, from: data)
        guard centres.schema == expectedSchema else {
            throw MotorControllerError.centresSchema(found: centres.schema, expected: expectedSchema)
        }
        return centres
    }

    /// The file bundled in the UEMAModel module's Resources folder.
    public static func bundled(in bundle: Bundle = UEMAModelModule.resourceBundle) throws -> RevivedCentres {
        guard let url = bundle.url(forResource: resourceName, withExtension: resourceExtension,
                                   subdirectory: resourceSubdirectory) else {
            throw MotorControllerError.centresNotFound(resource: resourceName + "." + resourceExtension)
        }
        return try decode(from: Data(contentsOf: url))
    }
}

// MARK: - Every number the controller reads

/// The released motor controller's numbers, read and shape-checked: the 25 motor_controller.*
/// tensors (520,812 parameters, receipt component_params.motor_controller), the fold constant,
/// and the two revived layers.
public struct MotorControllerParameters {
    /// 6 senses x 192 = 1,152 (motor_controller.py:146-147).
    public static let latentCount = SenseName.activeCount * Token.count
    /// 26 handles x 10 = 260 (motor_controller.py:159).
    public static let handleStateCount = Command.handleCount * 10
    /// 26 handles x 9 = 234 (motor_controller.py:62).
    public static let wellnessGradientCount = Command.count
    /// 1,152 + 260 + 234 = 1,646 (receipt motor_controller._input_dim).
    public static let inputCount = latentCount + handleStateCount + wellnessGradientCount
    /// hidden_dim 128 (receipt; lem_model.py LITE_OVERRIDES :282).
    public static let hiddenCount = 128
    /// confidence_hidden 128 (master_config.yaml:302).
    public static let confidenceHiddenCount = 128
    /// max_depth 4 (receipt; args.json:60 think_harder on).
    public static let blockCount = 4
    /// 26 x 9 = 234.
    public static let outputCount = Command.count

    /// The key prefix of the controller's tensors in model.safetensors.
    public static let prefix = "motor_controller."
    /// The fold's flag (motor_input_fold.py:124) and constant (motor_input_fold.py:95-104).
    public static let foldedKey = "motor_input_fold.folded"
    public static let foldConstantKey = "persistent_memory.readout.bias"
    public static let constantReturnKey = "persistent_memory.constant_return_updates"
    /// The repair's switch flag (persistent_memory.py:463); BOOL [] in model.safetensors, 1 in the
    /// released file.
    public static let repairSwitchedKey = "persistent_memory.repair_switched"
    /// The revived rows' keys (motor_input_fold.py:89, CONFIDENCE_ROWS_KEY; receipt
    /// liveness_repair.layers).
    public static let confidenceRowsKey = "motor_controller__confidence_head__0"
    public static let blockRowsKey = "motor_controller__residual_blocks__0__0"

    /// The 25 tensors of the manifest's motor_controller component with their shapes, in the
    /// manifest's key sequence; 520,812 elements in all.
    public static let tensorShapes: [(key: String, shape: [Int])] = {
        typealias P = MotorControllerParameters
        let hidden = P.hiddenCount
        let confidenceHidden = P.confidenceHiddenCount
        let scalar: [Int] = []
        var out: [(key: String, shape: [Int])] = [
            (key: "motor_controller.confidence_head.0.bias", shape: [confidenceHidden]),
            (key: "motor_controller.confidence_head.0.weight", shape: [confidenceHidden, P.latentCount]),
            (key: "motor_controller.confidence_head.2.bias", shape: [1]),
            (key: "motor_controller.confidence_head.2.weight", shape: [1, confidenceHidden]),
            (key: "motor_controller.depth_threshold_raw", shape: scalar),
            (key: "motor_controller.input_proj.bias", shape: [hidden]),
            (key: "motor_controller.input_proj.weight", shape: [hidden, P.inputCount]),
            (key: "motor_controller.output_proj.bias", shape: [P.outputCount]),
            (key: "motor_controller.output_proj.weight", shape: [P.outputCount, hidden]),
        ]
        for block in 0 ..< P.blockCount {
            out.append((key: "motor_controller.residual_blocks.\(block).0.bias", shape: [hidden]))
            out.append((key: "motor_controller.residual_blocks.\(block).0.weight", shape: [hidden, hidden]))
            out.append((key: "motor_controller.residual_blocks.\(block).2.bias", shape: [hidden]))
            out.append((key: "motor_controller.residual_blocks.\(block).2.weight", shape: [hidden, hidden]))
        }
        return out
    }()

    public let confidenceLayer0: DenseWeights
    public let confidenceLayer2: DenseWeights
    public let inputProjection: DenseWeights
    /// For each of the 4 blocks, its two Linear layers (indices 0 and 2 of the Sequential).
    public let blocks: [(first: DenseWeights, second: DenseWeights)]
    public let outputProjection: DenseWeights
    /// depth_threshold_raw: the logit of the learned depth threshold (motor_controller.py:80-82).
    public let depthThresholdRaw: Float
    /// The constant subtracted from the newest latents; nil when the controller is not folded.
    public let foldConstant: [Float]?
    /// The revived units of confidence_head.0; nil when none.
    public let revivedConfidence: RevivedRowsWeights?
    /// The revived units of residual_blocks.0.0; nil when none.
    public let revivedBlock0: RevivedRowsWeights?

    public init(confidenceLayer0: DenseWeights, confidenceLayer2: DenseWeights, inputProjection: DenseWeights,
                blocks: [(first: DenseWeights, second: DenseWeights)], outputProjection: DenseWeights,
                depthThresholdRaw: Float, foldConstant: [Float]?, revivedConfidence: RevivedRowsWeights?,
                revivedBlock0: RevivedRowsWeights?) throws {
        let p = MotorControllerParameters.self
        try p.checkDense(confidenceLayer0, input: p.latentCount, output: p.confidenceHiddenCount, name: "confidence_head.0")
        try p.checkDense(confidenceLayer2, input: p.confidenceHiddenCount, output: 1, name: "confidence_head.2")
        try p.checkDense(inputProjection, input: p.inputCount, output: p.hiddenCount, name: "input_proj")
        guard blocks.count == p.blockCount else {
            throw MotorControllerError.wrongInputCount(name: "residual_blocks", expected: p.blockCount, found: blocks.count)
        }
        for (index, block) in blocks.enumerated() {
            try p.checkDense(block.first, input: p.hiddenCount, output: p.hiddenCount, name: "residual_blocks.\(index).0")
            try p.checkDense(block.second, input: p.hiddenCount, output: p.hiddenCount, name: "residual_blocks.\(index).2")
        }
        try p.checkDense(outputProjection, input: p.hiddenCount, output: p.outputCount, name: "output_proj")
        if let constant = foldConstant, constant.count != p.latentCount {
            throw MotorControllerError.wrongInputCount(name: "fold constant", expected: p.latentCount, found: constant.count)
        }
        if let rows = revivedConfidence {
            try p.checkDense(rows.linear, input: p.latentCount, output: p.confidenceHiddenCount, name: rows.key)
        }
        if let rows = revivedBlock0 {
            try p.checkDense(rows.linear, input: p.hiddenCount, output: p.hiddenCount, name: rows.key)
        }
        self.confidenceLayer0 = confidenceLayer0
        self.confidenceLayer2 = confidenceLayer2
        self.inputProjection = inputProjection
        self.blocks = blocks
        self.outputProjection = outputProjection
        self.depthThresholdRaw = depthThresholdRaw
        self.foldConstant = foldConstant
        self.revivedConfidence = revivedConfidence
        self.revivedBlock0 = revivedBlock0
    }

    private static func checkDense(_ dense: DenseWeights, input: Int, output: Int, name: String) throws {
        guard dense.inputCount == input, dense.outputCount == output else {
            throw MotorControllerError.wrongShape(key: name, expected: [output, input],
                                                  found: [dense.outputCount, dense.inputCount])
        }
    }

    /// Reads the 25 controller tensors, the fold (when `motor_input_fold.folded` exists and
    /// reads 1) and the revived rows (when `liveness_repair.rows.<key>.mask` exists and has a 1),
    /// taking the revived layers' centres from `centres`. Throws naming the first tensor that is
    /// missing or has another shape.
    public static func read(from source: any TensorSource, centres: RevivedCentres?) throws -> MotorControllerParameters {
        let confidence0 = try DenseWeights.read(prefix: prefix + "confidence_head.0", inputCount: latentCount,
                                                outputCount: confidenceHiddenCount, from: source)
        let confidence2 = try DenseWeights.read(prefix: prefix + "confidence_head.2", inputCount: confidenceHiddenCount,
                                                outputCount: 1, from: source)
        let inputProjection = try DenseWeights.read(prefix: prefix + "input_proj", inputCount: inputCount,
                                                    outputCount: hiddenCount, from: source)
        var blocks: [(first: DenseWeights, second: DenseWeights)] = []
        for block in 0 ..< blockCount {
            let first = try DenseWeights.read(prefix: prefix + "residual_blocks.\(block).0", inputCount: hiddenCount,
                                              outputCount: hiddenCount, from: source)
            let second = try DenseWeights.read(prefix: prefix + "residual_blocks.\(block).2", inputCount: hiddenCount,
                                               outputCount: hiddenCount, from: source)
            blocks.append((first: first, second: second))
        }
        let outputProjection = try DenseWeights.read(prefix: prefix + "output_proj", inputCount: hiddenCount,
                                                     outputCount: outputCount, from: source)
        let threshold = try readTensor(prefix + "depth_threshold_raw", shape: [], from: source)

        var foldConstant: [Float]? = nil
        var folded = false
        if source.hasTensor(named: foldedKey) {
            folded = try readTensor(foldedKey, shape: [], from: source).values[0] != 0
        }
        if folded {
            // The return runs only when the repair has switched AND its length is above 0
            // (persistent_memory.py:463, `_return_on = repair_switched and
            // constant_return_updates > 0`); an absent repair_switched flag is read as switched,
            // so a file without it is refused on the length alone.
            var switched = true
            if source.hasTensor(named: repairSwitchedKey) {
                switched = try readTensor(repairSwitchedKey, shape: [], from: source).values[0] != 0
            }
            if switched, source.hasTensor(named: constantReturnKey) {
                let updates = try readTensor(constantReturnKey, shape: [], from: source).values[0]
                guard updates <= 0 else {
                    throw MotorControllerError.unsupportedConstantReturn(updates: updates)
                }
            }
            foldConstant = try readTensor(foldConstantKey, shape: [latentCount], from: source).values
        }

        let revivedConfidence = try RevivedRowsWeights.read(
            key: confidenceRowsKey, inputCount: latentCount, outputCount: confidenceHiddenCount,
            centre: centres?.confidenceHead0.centreMean, from: source)
        let revivedBlock0 = try RevivedRowsWeights.read(
            key: blockRowsKey, inputCount: hiddenCount, outputCount: hiddenCount,
            centre: centres?.residualBlock0.centreMean, from: source)

        return try MotorControllerParameters(
            confidenceLayer0: confidence0, confidenceLayer2: confidence2, inputProjection: inputProjection,
            blocks: blocks, outputProjection: outputProjection, depthThresholdRaw: threshold.values[0],
            foldConstant: foldConstant, revivedConfidence: revivedConfidence, revivedBlock0: revivedBlock0)
    }
}

// MARK: - Input and output

/// The controller's input for one model step: 1,152 + 260 + 234 = 1,646 numbers.
public struct MotorControllerInput: Equatable, Sendable {
    /// The forward model's output for the newest model step, 6 x 192 in the sequence vision,
    /// tactile, audio, proprioception, motor, vestibular (forward/last_step).
    public let newestLatents: [Float]
    /// 26 handles x 10: position cm / 200 (3), quaternion x y z w (4), commanded radius,
    /// commanded strength, mask (proprio_rows.md section 5).
    public let handleState: [Float]
    /// 26 handles x 9: d(wellness sum) / d(centred handle pose) (MotorStateBoundary.gradient).
    public let wellnessGradient: [Float]

    /// Checks the three counts and that every number is finite.
    public init(newestLatents: [Float], handleState: [Float], wellnessGradient: [Float]) throws {
        let p = MotorControllerParameters.self
        let parts: [(String, [Float], Int)] = [
            ("newest latents", newestLatents, p.latentCount),
            ("handle state", handleState, p.handleStateCount),
            ("wellness gradient", wellnessGradient, p.wellnessGradientCount),
        ]
        for (name, values, expected) in parts {
            guard values.count == expected else {
                throw MotorControllerError.wrongInputCount(name: name, expected: expected, found: values.count)
            }
            guard values.allSatisfy({ $0.isFinite }) else {
                throw MotorControllerError.nonFiniteInput(name)
            }
        }
        self.newestLatents = newestLatents
        self.handleState = handleState
        self.wellnessGradient = wellnessGradient
    }
}

/// The controller's output for one model step.
public struct MotorControllerOutput: Equatable, Sendable {
    /// The command, 26 body handles x 9 numbers, scale `.normalized`.
    public let command: Command
    /// confidence_head's sigmoid, 0 to 1.
    public let confidence: Float
    /// sigmoid(depth_threshold_raw), 0 to 1.
    public let depthThreshold: Float
    /// 1 + 3 sigmoid(5 (threshold - confidence)), 1 to 4.
    public let rawDepth: Float
    /// The four block weights sigmoid(5 (rawDepth - (i + 1))), each 0 to 1.
    public let blockWeights: [Float]
    /// max(|raw position| - 1, 0) per handle and axis, 78 numbers (motor_controller.py:236);
    /// 0 when the raw position lies inside -1 ... 1.
    public let positionExcursion: [Float]
}

/// Gives the command of one model step; the session (WI-32) and the forward model (WI-35)
/// depend on this protocol.
public protocol MotorCommanding {
    func command(for input: MotorControllerInput) throws -> MotorControllerOutput
}

// MARK: - The controller

/// The released motor controller (steps 1 to 7 of this file's header).
public struct MotorController: MotorCommanding {
    public let parameters: MotorControllerParameters
    private let confidenceLayer0: any AffineLayer
    private let confidenceLayer2: any AffineLayer
    private let inputProjection: any AffineLayer
    private let blocks: [(first: any AffineLayer, second: any AffineLayer)]
    private let outputProjection: any AffineLayer
    private let revivedConfidence: RevivedRows?
    private let revivedBlock0: RevivedRows?

    /// `builder` makes every Linear layer (MLXAffineLayerBuilder in the app).
    public init(parameters: MotorControllerParameters, builder: any AffineLayerBuilding) {
        self.parameters = parameters
        confidenceLayer0 = builder.makeLayer(parameters.confidenceLayer0)
        confidenceLayer2 = builder.makeLayer(parameters.confidenceLayer2)
        inputProjection = builder.makeLayer(parameters.inputProjection)
        var layers: [(first: any AffineLayer, second: any AffineLayer)] = []
        for block in parameters.blocks {
            layers.append((first: builder.makeLayer(block.first), second: builder.makeLayer(block.second)))
        }
        blocks = layers
        outputProjection = builder.makeLayer(parameters.outputProjection)
        revivedConfidence = parameters.revivedConfidence.map { RevivedRows(weights: $0, builder: builder) }
        revivedBlock0 = parameters.revivedBlock0.map { RevivedRows(weights: $0, builder: builder) }
    }

    /// Reads the parameters from `source` (LoadedWeights in the app) with the bundled centres.
    public static func released(from source: any TensorSource,
                                centres: RevivedCentres? = nil,
                                builder: any AffineLayerBuilding = MLXAffineLayerBuilder()) throws -> MotorController {
        let resolved = try centres ?? RevivedCentres.bundled()
        return MotorController(parameters: try MotorControllerParameters.read(from: source, centres: resolved),
                               builder: builder)
    }

    public func command(for input: MotorControllerInput) throws -> MotorControllerOutput {
        typealias M = MotorControllerMath

        // 1. the fold
        var read = input.newestLatents
        if let constant = parameters.foldConstant {
            for index in 0 ..< read.count {
                read[index] -= constant[index]
            }
        }

        // 2. confidence
        var confidencePre = confidenceLayer0.apply(read)
        if let rows = revivedConfidence {
            confidencePre = rows.apply(input: read, preActivation: confidencePre)
        }
        let confidenceHidden = confidencePre.map { Float(M.gelu(Double($0))) }
        let confidence = M.sigmoid(Double(confidenceLayer2.apply(confidenceHidden)[0]))

        // 3. the 1,646 numbers
        let x = read + input.handleState + input.wellnessGradient

        // 4. depth
        let threshold = M.sigmoid(Double(parameters.depthThresholdRaw))
        let maxDepth = Double(MotorControllerParameters.blockCount)
        let rawDepth = 1.0 + (maxDepth - 1.0) * M.sigmoid(5.0 * (threshold - confidence))

        // 5. input projection and the four gated blocks
        var h = inputProjection.apply(x)
        var blockWeights: [Float] = []
        for (index, block) in blocks.enumerated() {
            let weight = Float(M.sigmoid(5.0 * (rawDepth - Double(index + 1))))
            blockWeights.append(weight)
            var pre = block.first.apply(h)
            if index == 0, let rows = revivedBlock0 {
                pre = rows.apply(input: h, preActivation: pre)
            }
            let inner = block.second.apply(pre.map { Float(M.gelu(Double($0))) })
            for unit in 0 ..< h.count {
                h[unit] = weight * inner[unit] + h[unit]
            }
        }

        // 6. output projection, 7. per-row activations
        let raw = outputProjection.apply(h)
        var values = [Float](repeating: 0, count: Command.count)
        var excursion = [Float](repeating: 0, count: Command.handleCount * 3)
        for handle in 0 ..< Command.handleCount {
            let base = handle * Command.numbersPerHandle
            for axis in 0 ..< 3 {
                let value = raw[base + axis]
                values[base + axis] = min(max(value, -1), 1)
                excursion[handle * 3 + axis] = max(abs(value) - 1, 0)
            }
            var norm = 0.0
            for column in 3 ..< 7 {
                norm += Double(raw[base + column]) * Double(raw[base + column])
            }
            let divisor = max(norm.squareRoot(), 1e-8)
            for column in 3 ..< 7 {
                values[base + column] = Float(Double(raw[base + column]) / divisor)
            }
            values[base + 7] = Float(M.sigmoid(Double(raw[base + 7])))
            values[base + 8] = Float(M.sigmoid(Double(raw[base + 8])))
        }

        return MotorControllerOutput(
            command: try Command(values: values, scale: .normalized),
            confidence: Float(confidence),
            depthThreshold: Float(threshold),
            rawDepth: Float(rawDepth),
            blockWeights: blockWeights,
            positionExcursion: excursion)
    }
}
