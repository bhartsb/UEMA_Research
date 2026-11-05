// MotorControllerTests.swift: the tests of Sources/UEMAModel/MotorController.swift (work item
// WI-36). Nothing in this file was compiled on the machine that wrote it (no Swift toolchain on
// Windows). Every test but the last runs without model.safetensors; the last reads it and the
// three model-pass fixtures of WI-34 and skips when the file is absent (the same rule as
// SafetensorsLoaderTests: UEMA_WEIGHTS_PATH, else the manifest's source.path under the repository
// root). The pass line of the fixture test, 1e-4 absolute on each of the 234 command numbers, and
// the measured 2.38e-6 of the float32 torch reproduction it rests on are in MotorController.swift's
// header.

import XCTest
import Foundation
import MLX
import UEMACore
@testable import UEMAModel

/// A fixture tensor the test needs is not in the file.
struct ControllerFixtureMissingTensor: Error, CustomStringConvertible {
    let fixture: String
    let key: String
    var description: String { "fixture \(fixture) has no tensor \(key)" }
}

/// One model-pass fixture of WI-34 (Tests/UEMAModelTests/Fixtures/model_pass/<name>.safetensors,
/// every tensor F32, so MLX's own safetensors reader accepts it), read with MLX.loadArrays(url:)
/// (Source/MLX/IO.swift, quoted in SafetensorsLoader.swift's header). Named for this work item's
/// two test classes (MotorControllerTests, WellnessTests), so that it does not collide with a
/// fixture reader another test file of the UEMAModelTests target declares.
struct ControllerFixture {
    /// The three states of the fixtures: the adult absent, presenting and engaged.
    static let names = ["step_000492800", "step_000493674", "step_000493980"]

    let name: String
    let arrays: [String: MLXArray]

    init(name: String) throws {
        guard let url = Bundle.module.url(forResource: name, withExtension: "safetensors",
                                          subdirectory: "Fixtures/model_pass") else {
            throw XCTSkip("fixture \(name).safetensors is not in the test bundle; run apple/scripts/export_model_pass_fixtures.py")
        }
        self.name = name
        self.arrays = try MLX.loadArrays(url: url)
    }

    /// True when the fixture holds a tensor of `key`.
    func has(_ key: String) -> Bool {
        arrays[key] != nil
    }

    /// The float32 numbers of one tensor, row-major; throws naming the key when it is absent.
    func floats(_ key: String) throws -> [Float] {
        guard let array = arrays[key] else {
            throw ControllerFixtureMissingTensor(fixture: name, key: key)
        }
        return array.asType(.float32).asArray(Float.self)
    }

    /// One 192-number token.
    func token(_ key: String) throws -> Token {
        try Token(values: floats(key))
    }
}

/// The released weights file, or nil when it is not on this machine.
func controllerWeightsURLIfPresent(manifest: WeightManifest, filePath: String = #filePath) -> URL? {
    let url: URL
    if let override = ProcessInfo.processInfo.environment["UEMA_WEIGHTS_PATH"], !override.isEmpty {
        url = URL(fileURLWithPath: override)
    } else {
        var root = URL(fileURLWithPath: filePath)
        for _ in 0 ..< 5 {
            root = root.deletingLastPathComponent()
        }
        url = root.appendingPathComponent(manifest.source.path)
    }
    return FileManager.default.fileExists(atPath: url.path) ? url : nil
}

final class MotorControllerTests: XCTestCase {

    // MARK: Synthetic tensors

    /// The 25 controller tensors, every number 0 except those `fill` sets.
    private func zeroTensors() -> [String: ShapedArray] {
        var tensors: [String: ShapedArray] = [:]
        for entry in MotorControllerParameters.tensorShapes {
            tensors[entry.key] = ShapedArray(zerosOfShape: entry.shape)
        }
        return tensors
    }

    private func setValue(_ tensors: inout [String: ShapedArray], _ key: String, index: Int, to value: Float) throws {
        guard let tensor = tensors[key] else {
            XCTFail("no tensor \(key)")
            return
        }
        var values = tensor.values
        values[index] = value
        tensors[key] = try ShapedArray(shape: tensor.shape, values: values)
    }

    private func zeroInput(latent0: Float = 0) throws -> MotorControllerInput {
        var latents = [Float](repeating: 0, count: MotorControllerParameters.latentCount)
        latents[0] = latent0
        return try MotorControllerInput(newestLatents: latents,
                                        handleState: [Float](repeating: 0, count: MotorControllerParameters.handleStateCount),
                                        wellnessGradient: [Float](repeating: 0, count: MotorControllerParameters.wellnessGradientCount))
    }

    // MARK: Shapes against the manifest

    /// The 25 keys and shapes the controller reads equal the manifest's motor_controller
    /// component, 520,812 elements (receipt component_params.motor_controller).
    func testTensorTableEqualsTheManifest() throws {
        let manifest = try WeightManifest.bundled()
        let records = manifest.tensors(inComponent: "motor_controller")
        var manifestShapes: [String: [Int]] = [:]
        for record in records {
            manifestShapes[record.key] = record.shape
        }
        var portShapes: [String: [Int]] = [:]
        for entry in MotorControllerParameters.tensorShapes {
            portShapes[entry.key] = entry.shape
        }
        XCTAssertEqual(portShapes, manifestShapes)
        XCTAssertEqual(MotorControllerParameters.tensorShapes.count, 25)
        let elements = MotorControllerParameters.tensorShapes.reduce(0) { $0 + $1.shape.reduce(1, *) }
        XCTAssertEqual(elements, 520_812)
        XCTAssertEqual(MotorControllerParameters.inputCount, 1_646)
    }

    /// The fold and revived-row tensors the controller also reads are in the manifest with the
    /// shapes it checks.
    func testHookTensorsAreInTheManifest() throws {
        let manifest = try WeightManifest.bundled()
        let expected: [String: [Int]] = [
            MotorControllerParameters.foldedKey: [],
            MotorControllerParameters.foldConstantKey: [1_152],
            MotorControllerParameters.constantReturnKey: [],
            "liveness_repair.rows.motor_controller__confidence_head__0.mask": [128],
            "liveness_repair.rows.motor_controller__confidence_head__0.norm.weight": [1_152],
            "liveness_repair.rows.motor_controller__confidence_head__0.lin.weight": [128, 1_152],
            "liveness_repair.rows.motor_controller__residual_blocks__0__0.mask": [128],
            "liveness_repair.rows.motor_controller__residual_blocks__0__0.norm.bias": [128],
            "liveness_repair.rows.motor_controller__residual_blocks__0__0.lin.weight": [128, 128],
        ]
        for (key, shape) in expected {
            XCTAssertEqual(manifest.byKey[key]?.shape, shape, key)
        }
    }

    /// The bundled centres hold 1,152 and 128 finite numbers under the rows' keys.
    func testBundledCentresDecode() throws {
        let centres = try RevivedCentres.bundled()
        XCTAssertEqual(centres.confidenceHead0.key, MotorControllerParameters.confidenceRowsKey)
        XCTAssertEqual(centres.residualBlock0.key, MotorControllerParameters.blockRowsKey)
        XCTAssertEqual(centres.confidenceHead0.centreMean.count, 1_152)
        XCTAssertEqual(centres.residualBlock0.centreMean.count, 128)
        XCTAssertTrue(centres.confidenceHead0.centreMean.allSatisfy { $0.isFinite })
        XCTAssertTrue(centres.residualBlock0.centreMean.allSatisfy { $0.isFinite })
    }

    func testOtherCentresSchemaIsRefused() throws {
        let json = #"{"schema": "other", "confidence_head_0": {"key": "a", "centre_weight": 1, "centre_mean": []}, "residual_blocks_0_0": {"key": "b", "centre_weight": 1, "centre_mean": []}}"#
        XCTAssertThrowsError(try RevivedCentres.decode(from: Data(json.utf8))) { error in
            XCTAssertEqual(error as? MotorControllerError,
                           .centresSchema(found: "other", expected: RevivedCentres.expectedSchema))
        }
    }

    // MARK: The network on hand-set numbers

    /// All weights 0: confidence = sigmoid(confidence_head.2.bias), h = input_proj.bias passes the
    /// four blocks unchanged (their biases are 0), the command is the output bias after the
    /// per-row activations: position clamped to -1 ... 1 with its excursion, rotation divided by
    /// its length, radius and strength through the sigmoid.
    func testZeroWeightsGiveTheActivatedOutputBias() throws {
        var tensors = zeroTensors()
        try setValue(&tensors, "motor_controller.confidence_head.2.bias", index: 0, to: 2)
        // handle 0: position (3, -0.5, 0), rotation (0, 0, 3, 4), radius logit 0, strength logit ln 3
        let row: [Float] = [3, -0.5, 0, 0, 0, 3, 4, 0, Float(log(3.0))]
        for (column, value) in row.enumerated() {
            try setValue(&tensors, "motor_controller.output_proj.bias", index: column, to: value)
        }
        let parameters = try MotorControllerParameters.read(from: DictionaryTensorSource(tensors: tensors), centres: nil)
        XCTAssertNil(parameters.foldConstant)
        XCTAssertNil(parameters.revivedConfidence)
        let controller = MotorController(parameters: parameters, builder: ReferenceAffineLayerBuilder())
        let output = try controller.command(for: zeroInput())

        let confidence = 1 / (1 + exp(-2.0))
        XCTAssertEqual(Double(output.confidence), confidence, accuracy: 1e-6)
        XCTAssertEqual(Double(output.depthThreshold), 0.5, accuracy: 1e-7)
        let rawDepth = 1 + 3 / (1 + exp(-5 * (0.5 - confidence)))
        XCTAssertEqual(Double(output.rawDepth), rawDepth, accuracy: 1e-5)
        XCTAssertEqual(output.blockWeights.count, 4)

        let handle0 = output.command.row(.wristL)
        XCTAssertEqual(handle0[0], 1, accuracy: 1e-7)
        XCTAssertEqual(handle0[1], -0.5, accuracy: 1e-7)
        XCTAssertEqual(handle0[2], 0, accuracy: 1e-7)
        XCTAssertEqual(handle0[5], 0.6, accuracy: 1e-6)
        XCTAssertEqual(handle0[6], 0.8, accuracy: 1e-6)
        XCTAssertEqual(handle0[7], 0.5, accuracy: 1e-6)
        XCTAssertEqual(handle0[8], 0.75, accuracy: 1e-6)
        XCTAssertEqual(output.positionExcursion[0], 2, accuracy: 1e-6)
        XCTAssertEqual(output.positionExcursion[1], 0)
        // handle 1: rotation all 0, divided by max(0, 1e-8): 0
        XCTAssertEqual(output.command.row(.thumbL)[3], 0)
        XCTAssertEqual(output.command.scale, .normalized)
    }

    /// The fold subtracts persistent_memory.readout.bias from the newest latents before the
    /// input projection: with input_proj reading latent 0 into hidden unit 0 and output_proj
    /// copying hidden unit 0 to the first position number, latent 0.5 minus constant 0.25 gives
    /// 0.25; without the folded flag, 0.5.
    func testFoldSubtractsTheMemoryConstant() throws {
        var tensors = zeroTensors()
        try setValue(&tensors, "motor_controller.input_proj.weight", index: 0, to: 1)
        try setValue(&tensors, "motor_controller.output_proj.weight", index: 0, to: 1)
        var constant = [Float](repeating: 0, count: 1_152)
        constant[0] = 0.25
        tensors[MotorControllerParameters.foldConstantKey] = try ShapedArray(shape: [1_152], values: constant)
        tensors[MotorControllerParameters.constantReturnKey] = try ShapedArray(shape: [], values: [0])

        tensors[MotorControllerParameters.foldedKey] = try ShapedArray(shape: [], values: [1])
        let folded = MotorController(parameters: try MotorControllerParameters.read(
            from: DictionaryTensorSource(tensors: tensors), centres: nil), builder: ReferenceAffineLayerBuilder())
        XCTAssertEqual(try folded.command(for: zeroInput(latent0: 0.5)).command.values[0], 0.25, accuracy: 1e-7)

        tensors[MotorControllerParameters.foldedKey] = try ShapedArray(shape: [], values: [0])
        let unfolded = MotorController(parameters: try MotorControllerParameters.read(
            from: DictionaryTensorSource(tensors: tensors), centres: nil), builder: ReferenceAffineLayerBuilder())
        XCTAssertEqual(try unfolded.command(for: zeroInput(latent0: 0.5)).command.values[0], 0.5, accuracy: 1e-7)
    }

    /// A running constant return (constant_return_updates > 0) is refused: its factor is not
    /// computed by this port.
    func testConstantReturnIsRefused() throws {
        var tensors = zeroTensors()
        tensors[MotorControllerParameters.foldedKey] = try ShapedArray(shape: [], values: [1])
        tensors[MotorControllerParameters.foldConstantKey] = ShapedArray(zerosOfShape: [1_152])
        tensors[MotorControllerParameters.constantReturnKey] = try ShapedArray(shape: [], values: [500])
        XCTAssertThrowsError(try MotorControllerParameters.read(from: DictionaryTensorSource(tensors: tensors), centres: nil)) { error in
            XCTAssertEqual(error as? MotorControllerError, .unsupportedConstantReturn(updates: 500))
        }
    }

    /// Before the repair's switch (repair_switched 0) the return is off whatever its length
    /// (persistent_memory.py:463), so the same file is read and the fold constant is the readout
    /// bias.
    func testConstantReturnBeforeTheSwitchIsAccepted() throws {
        var tensors = zeroTensors()
        tensors[MotorControllerParameters.foldedKey] = try ShapedArray(shape: [], values: [1])
        tensors[MotorControllerParameters.foldConstantKey] = ShapedArray(zerosOfShape: [1_152])
        tensors[MotorControllerParameters.constantReturnKey] = try ShapedArray(shape: [], values: [500])
        tensors[MotorControllerParameters.repairSwitchedKey] = try ShapedArray(shape: [], values: [0])
        let parameters = try MotorControllerParameters.read(from: DictionaryTensorSource(tensors: tensors), centres: nil)
        XCTAssertEqual(parameters.foldConstant?.count, 1_152)

        tensors[MotorControllerParameters.repairSwitchedKey] = try ShapedArray(shape: [], values: [1])
        XCTAssertThrowsError(try MotorControllerParameters.read(from: DictionaryTensorSource(tensors: tensors), centres: nil)) { error in
            XCTAssertEqual(error as? MotorControllerError, .unsupportedConstantReturn(updates: 500))
        }
    }

    /// The revived units of residual_blocks.0.0 enter the gated residual path of step 5. Hand-set
    /// numbers: input_proj.bias[0] 2, so h = (2, 0, ..., 0) (128 numbers); block 0's first Linear
    /// is all 0, its revived unit 0 reads lin row 0 = (1, 0, ..., 0) of LayerNorm(h - 0) with
    /// weight 1 and bias 0; block 0's second Linear copies unit 0; output_proj.weight[0][0] 0.01.
    /// Confidence 0.5 and threshold 0.5 give raw depth 2.5 and block 0 weight sigmoid(7.5). So the
    /// first position number is 0.01 (sigmoid(7.5) gelu(n0) + 2), n0 = (2 - 2/128) /
    /// sqrt(var + 1e-5), var the mean of the 128 squared deviations from 2/128; without the revived
    /// rows it is 0.01 x 2 = 0.02.
    func testRevivedBlockUnitsEnterTheResidualPath() throws {
        var tensors = zeroTensors()
        try setValue(&tensors, "motor_controller.input_proj.bias", index: 0, to: 2)
        try setValue(&tensors, "motor_controller.residual_blocks.0.2.weight", index: 0, to: 1)
        try setValue(&tensors, "motor_controller.output_proj.weight", index: 0, to: 0.01)
        let plain = try MotorControllerParameters.read(from: DictionaryTensorSource(tensors: tensors), centres: nil)
        XCTAssertNil(plain.revivedBlock0)
        let plainOutput = try MotorController(parameters: plain, builder: ReferenceAffineLayerBuilder())
            .command(for: zeroInput())
        XCTAssertEqual(plainOutput.command.values[0], 0.02, accuracy: 1e-6)

        let hidden = MotorControllerParameters.hiddenCount
        var linWeight = [Float](repeating: 0, count: hidden * hidden)
        linWeight[0] = 1
        var mask = [Bool](repeating: false, count: hidden)
        mask[0] = true
        let rows = try RevivedRowsWeights(
            key: MotorControllerParameters.blockRowsKey, mask: mask,
            normWeight: [Float](repeating: 1, count: hidden), normBias: [Float](repeating: 0, count: hidden),
            linear: try DenseWeights(inputCount: hidden, outputCount: hidden, weight: linWeight,
                                     bias: [Float](repeating: 0, count: hidden)),
            centre: [Float](repeating: 0, count: hidden))
        let revived = try MotorControllerParameters(
            confidenceLayer0: plain.confidenceLayer0, confidenceLayer2: plain.confidenceLayer2,
            inputProjection: plain.inputProjection, blocks: plain.blocks, outputProjection: plain.outputProjection,
            depthThresholdRaw: plain.depthThresholdRaw, foldConstant: nil, revivedConfidence: nil,
            revivedBlock0: rows)
        let output = try MotorController(parameters: revived, builder: ReferenceAffineLayerBuilder())
            .command(for: zeroInput())

        let mean = 2.0 / Double(hidden)
        let variance = ((2.0 - mean) * (2.0 - mean) + Double(hidden - 1) * mean * mean) / Double(hidden)
        let n0 = (2.0 - mean) / (variance + 1e-5).squareRoot()
        let gelu = 0.5 * n0 * (1.0 + erf(n0 / 2.0.squareRoot()))
        let weight0 = 1.0 / (1.0 + exp(-7.5))
        XCTAssertEqual(Double(output.rawDepth), 2.5, accuracy: 1e-6)
        XCTAssertEqual(Double(output.blockWeights[0]), weight0, accuracy: 1e-6)
        XCTAssertEqual(Double(output.command.values[0]), 0.01 * (weight0 * gelu + 2.0), accuracy: 1e-5)
        XCTAssertEqual(output.positionExcursion[0], 0)
    }

    /// Revived rows with a revived unit and no centre are refused, naming the layer.
    func testRevivedRowsWithoutCentreAreRefused() throws {
        var tensors = zeroTensors()
        let prefix = "liveness_repair.rows." + MotorControllerParameters.blockRowsKey
        var mask = [Float](repeating: 0, count: 128)
        mask[3] = 1
        tensors[prefix + ".mask"] = try ShapedArray(shape: [128], values: mask)
        tensors[prefix + ".norm.weight"] = ShapedArray(zerosOfShape: [128])
        tensors[prefix + ".norm.bias"] = ShapedArray(zerosOfShape: [128])
        tensors[prefix + ".lin.weight"] = ShapedArray(zerosOfShape: [128, 128])
        tensors[prefix + ".lin.bias"] = ShapedArray(zerosOfShape: [128])
        XCTAssertThrowsError(try MotorControllerParameters.read(from: DictionaryTensorSource(tensors: tensors), centres: nil)) { error in
            XCTAssertEqual(error as? MotorControllerError, .missingCentre(layer: MotorControllerParameters.blockRowsKey))
        }
    }

    /// RevivedRows replaces only the masked units: input (1, 2, 3, 4), centre 0, LayerNorm weight
    /// 1 and bias 0 give (x - 2.5) / sqrt(1.25 + 1e-5); lin row 0 = (1, 0, 0, 0) reads its first
    /// number, -1.5 / 1.1180384 = -1.3416363; unit 1 keeps the layer's own 20.
    func testRevivedRowsReplaceOnlyMaskedUnits() throws {
        let linear = try DenseWeights(inputCount: 4, outputCount: 2,
                                      weight: [1, 0, 0, 0, 0, 1, 0, 0], bias: [0, 0])
        let weights = try RevivedRowsWeights(key: "test", mask: [true, false], normWeight: [1, 1, 1, 1],
                                             normBias: [0, 0, 0, 0], linear: linear, centre: [0, 0, 0, 0])
        XCTAssertEqual(weights.revivedCount, 1)
        let rows = RevivedRows(weights: weights, builder: ReferenceAffineLayerBuilder())
        let out = rows.apply(input: [1, 2, 3, 4], preActivation: [10, 20])
        XCTAssertEqual(Double(out[0]), -1.5 / (1.25 + 1e-5).squareRoot(), accuracy: 1e-6)
        XCTAssertEqual(out[1], 20)
    }

    /// Wrong input counts and non-finite numbers are refused.
    func testInputIsChecked() {
        XCTAssertThrowsError(try MotorControllerInput(newestLatents: [0], handleState: [], wellnessGradient: []))
        var latents = [Float](repeating: 0, count: 1_152)
        latents[7] = .nan
        XCTAssertThrowsError(try MotorControllerInput(
            newestLatents: latents, handleState: [Float](repeating: 0, count: 260),
            wellnessGradient: [Float](repeating: 0, count: 234))) { error in
            XCTAssertEqual(error as? MotorControllerError, .nonFiniteInput("newest latents"))
        }
    }

    /// The MLX layer and the plain-Swift layer agree on a 12 x 7 layer to 1e-5.
    func testMLXLayerMatchesReferenceLayer() throws {
        let inputCount = 7
        let outputCount = 12
        var weight: [Float] = []
        for index in 0 ..< inputCount * outputCount {
            weight.append(Float(sin(Double(index) * 0.37)))
        }
        let bias = (0 ..< outputCount).map { Float(cos(Double($0) * 0.91)) }
        let dense = try DenseWeights(inputCount: inputCount, outputCount: outputCount, weight: weight, bias: bias)
        let input = (0 ..< inputCount).map { Float(Double($0) * 0.5 - 1.25) }
        let viaMLX = MLXAffineLayer(dense).apply(input)
        let viaSwift = ReferenceAffineLayer(dense).apply(input)
        XCTAssertEqual(viaMLX.count, outputCount)
        for index in 0 ..< outputCount {
            XCTAssertEqual(viaMLX[index], viaSwift[index], accuracy: 1e-5, "output \(index)")
        }
    }

    // MARK: With the released file

    /// With model.safetensors: the controller reads the fold and both revived layers (84 and 127
    /// revived units, receipt liveness_repair.revived) and reproduces command/motor_out and
    /// command/confidence of the three fixtures from forward/last_step, command/handle_state and
    /// command/wellness_grad within 1e-4 per number, on MLX and on the plain-Swift layers.
    func testReleasedControllerReproducesTheFixtures() throws {
        let manifest = try WeightManifest.bundled()
        guard let url = controllerWeightsURLIfPresent(manifest: manifest) else {
            throw XCTSkip("model.safetensors is not on this machine; run fetch_weights.py --accept-license or set UEMA_WEIGHTS_PATH")
        }
        let loaded = try SafetensorsLoader(manifest: manifest).load(url: url)
        let centres = try RevivedCentres.bundled()
        let parameters = try MotorControllerParameters.read(from: loaded, centres: centres)
        XCTAssertNotNil(parameters.foldConstant)
        XCTAssertEqual(parameters.revivedConfidence?.revivedCount, 84)
        XCTAssertEqual(parameters.revivedBlock0?.revivedCount, 127)

        let viaMLX = MotorController(parameters: parameters, builder: MLXAffineLayerBuilder())
        let viaSwift = MotorController(parameters: parameters, builder: ReferenceAffineLayerBuilder())
        let tolerance: Float = 1e-4
        for name in ControllerFixture.names {
            let fixture = try ControllerFixture(name: name)
            let input = try MotorControllerInput(newestLatents: fixture.floats("forward/last_step"),
                                                 handleState: fixture.floats("command/handle_state"),
                                                 wellnessGradient: fixture.floats("command/wellness_grad"))
            let expected = try fixture.floats("command/motor_out")
            let expectedConfidence = try fixture.floats("command/confidence")
            XCTAssertEqual(expected.count, Command.count)
            var largest: [Float] = []
            for controller in [viaMLX, viaSwift] {
                let output = try controller.command(for: input)
                var worst: Float = 0
                for index in 0 ..< Command.count {
                    worst = max(worst, abs(output.command.values[index] - expected[index]))
                }
                largest.append(worst)
                XCTAssertLessThanOrEqual(worst, tolerance, "\(name)")
                XCTAssertEqual(output.confidence, expectedConfidence.first ?? -1, accuracy: tolerance, "\(name)")
            }
            print("motor controller \(name): largest |difference| \(largest[0]) over 234 numbers (MLX), \(largest[1]) (reference)")
        }
    }
}
