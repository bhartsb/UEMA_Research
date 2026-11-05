// ModelPassTests.swift: the tests of work item WI-35 (Sources/UEMAModel/TransformerCore.swift,
// Encoders/*.swift, PredictionHeads.swift, ForwardModel.swift). Nothing in this file was compiled on
// the machine that wrote it (no Swift toolchain on Windows).
//
// WITHOUT model.safetensors (the file is not in git): every component is built from a synthetic
// weight source holding the manifest's keys and shapes (deterministic numbers sin(i) x 0.02, masks
// alternating true and false); the tests check that each component reads exactly its manifest
// component's parameter keys, that its parameter count equals the receipt's, that one forward pass
// gives the documented shapes with finite numbers, and the context buffer's and causal mask's rules.
// WITH model.safetensors (found at the manifest's source.path under the repository root, or at the
// environment variable UEMA_WEIGHTS_PATH; skipped when absent): the three WI-34 fixtures
// (Fixtures/model_pass/step_000492800, step_000493674, step_000493980) run through
// ForwardModel.released on the default MLX device (the Metal GPU on a Mac), with the fixture's
// forward/memory_bias handed in (the persistent memory is not ported); the test prints, per sense, the
// largest absolute difference of the pooled vector, the token and the predicted token, and fails when
// a token or predicted token differs by more than ModelPassTolerance.maxAbsoluteDifference (1e-3).
// The printed numbers go into the plan's section 9 row of WI-35.
//
// MLX symbols used here beyond those cited in TransformerCore.swift: loadArrays(url:stream:)
// (Source/MLX/IO.swift, quoted in SafetensorsLoader.swift's header) to read a fixture file.

import Foundation
import MLX
import UEMACore
import XCTest
@testable import UEMAModel

/// A weight source that records every key read (to compare a component's reads with the manifest).
final class RecordingTensorSource: MLXTensorSource {
    let base: any MLXTensorSource
    private(set) var readKeys: Set<String> = []

    init(base: any MLXTensorSource) {
        self.base = base
    }

    func tensor(_ key: String) throws -> MLXArray {
        let array = try base.tensor(key)
        readKeys.insert(key)
        return array
    }
}

/// Synthetic weights of the manifest's keys and shapes, built on first use per key.
final class SyntheticTensorSource: MLXTensorSource {
    let records: [String: ManifestTensor]
    private var cache: [String: MLXArray] = [:]

    init(manifest: WeightManifest) {
        self.records = manifest.byKey
    }

    func tensor(_ key: String) throws -> MLXArray {
        if let cached = cache[key] {
            return cached
        }
        guard let record = records[key] else {
            throw ModelPortError.missingTensor(key: key)
        }
        let count = record.shape.reduce(1, *)
        let array: MLXArray
        switch record.dtype {
        case .bool:
            array = MLXArray((0 ..< count).map { $0 % 2 == 0 }, record.shape)
        default:
            // deterministic numbers in -0.02 ... 0.02; float32 whatever the stored type (the port reads
            // only float32 tensors and boolean masks)
            array = MLXArray((0 ..< count).map { Float(sin(Double($0) * 0.7 + 0.3)) * 0.02 }, record.shape)
        }
        cache[key] = array
        return array
    }
}

final class ModelPassTests: XCTestCase {
    /// Before any MLX operation of the process (TransformerCore.swift, MLXFloat32Precision).
    override class func setUp() {
        super.setUp()
        print("MLX_ENABLE_TF32 in force: \(MLXFloat32Precision.enforce())")
    }


    // MARK: Helpers

    private func manifest() throws -> WeightManifest {
        try WeightManifest.bundled()
    }

    /// The parameter keys of one manifest component (buffers excluded).
    private func parameterKeys(_ manifest: WeightManifest, component: String) -> Set<String> {
        Set(manifest.tensors(inComponent: component).filter { $0.kind == .parameter }.map { $0.key })
    }

    /// The keys of `component`'s parameters read through `recorder`.
    private func readParameterKeys(_ recorder: RecordingTensorSource, manifest: WeightManifest,
                                   component: String) -> Set<String> {
        recorder.readKeys.filter { manifest.byKey[$0]?.component == component && manifest.byKey[$0]?.kind == .parameter }
    }

    private func randomInput(_ sense: SenseName, shape: [Int], auxiliary: [Int]? = nil) throws -> EncoderInput {
        let count = shape.reduce(1, *)
        let primary = try ShapedArray(shape: shape, values: (0 ..< count).map { Float(cos(Double($0) * 0.37)) * 0.5 })
        var aux: ShapedArray? = nil
        if let auxiliary {
            let auxCount = auxiliary.reduce(1, *)
            aux = try ShapedArray(shape: auxiliary, values: (0 ..< auxCount).map { Float(sin(Double($0) * 0.11)) })
        }
        return EncoderInput(sense: sense, primary: primary, auxiliary: aux)
    }

    private func syntheticInputs(patches: Int = 70) throws -> [SenseName: EncoderInput] {
        [
            .vision: try randomInput(.vision, shape: [patches, VisionEncoder.Released.patchWidth],
                                     auxiliary: [patches, VisionEncoder.Released.angleCount]),
            .tactile: try randomInput(.tactile, shape: [TactileEncoder.points, TactileEncoder.numbersPerPoint]),
            .audio: try randomInput(.audio, shape: [AudioEncoder.bins, AudioEncoder.ears],
                                    auxiliary: [AudioEncoder.spatialNumbers]),
            .proprioception: try randomInput(.proprioception, shape: [ProprioceptionEncoder.inputWidth]),
            .motor: try randomInput(.motor, shape: [MotorEncoder.handles, MotorEncoder.numbersPerRow]),
            .vestibular: try randomInput(.vestibular, shape: [VestibularEncoder.inputWidth]),
        ]
    }

    private func maxAbsoluteDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = NeuralOps.floats(a)
        let y = NeuralOps.floats(b)
        XCTAssertEqual(x.count, y.count, "compared arrays hold different counts")
        var largest: Float = 0
        for index in 0 ..< min(x.count, y.count) {
            largest = max(largest, abs(x[index] - y[index]))
        }
        return largest
    }

    // MARK: Without the weights

    /// Each encoder reads exactly its manifest component's parameter keys and has the receipt's
    /// parameter count: vision 2,106,048; tactile 1,074,816; audio 450,432; proprioception 171,648;
    /// motor 1,079,808; vestibular 117,888.
    func testEncodersReadTheirManifestKeysAndCounts() throws {
        let manifest = try manifest()
        let recorder = RecordingTensorSource(base: SyntheticTensorSource(manifest: manifest))
        func scope(_ sense: SenseName) -> ScopedMLXTensorSource {
            recorder.scoped("encoder_registry.encoders.\(sense.rawValue).")
        }
        let encoders: [(String, any SenseEncoding, Int)] = [
            ("enc_vision", try VisionEncoder(source: scope(.vision)), VisionEncoder.Released.parameterCount),
            ("enc_tactile", try TactileEncoder(source: scope(.tactile)), TactileEncoder.parameterCount),
            ("enc_audio", try AudioEncoder(source: scope(.audio)), AudioEncoder.parameterCount),
            ("enc_proprioception", try ProprioceptionEncoder(source: scope(.proprioception)),
             ProprioceptionEncoder.parameterCount),
            ("enc_motor", try MotorEncoder(source: scope(.motor)), MotorEncoder.parameterCount),
            ("enc_vestibular", try VestibularEncoder(source: scope(.vestibular)), VestibularEncoder.parameterCount),
        ]
        for (component, encoder, expected) in encoders {
            XCTAssertEqual(encoder.parameterCount, expected, component)
            XCTAssertEqual(encoder.parameterCount, manifest.components[component]?.parameterElements, component)
            XCTAssertEqual(readParameterKeys(recorder, manifest: manifest, component: component),
                           parameterKeys(manifest, component: component), component)
        }
    }

    /// The transformer reads every forward_model parameter key and counts 2,994,432; the prediction
    /// heads read every prediction_heads parameter key and count 1,114,560; the all-slot path counts
    /// 403,200.
    func testTransformerAndHeadsReadTheirManifestKeysAndCounts() throws {
        let manifest = try manifest()
        let recorder = RecordingTensorSource(base: SyntheticTensorSource(manifest: manifest))
        let core = try TransformerCore(source: recorder.scoped("forward_model."))
        XCTAssertEqual(core.parameterCount, TransformerCore.Released.parameterCount)
        XCTAssertEqual(readParameterKeys(recorder, manifest: manifest, component: "forward_model"),
                       parameterKeys(manifest, component: "forward_model"))
        let heads = try PredictionHeads(source: recorder.scoped("prediction_heads."))
        XCTAssertEqual(heads.parameterCount, PredictionHeads.parameterCount)
        XCTAssertEqual(readParameterKeys(recorder, manifest: manifest, component: "prediction_heads"),
                       parameterKeys(manifest, component: "prediction_heads"))
        let path = try HeadsAllSlotPath.released(from: recorder, centres: try PredictionHeadCentres.bundled())
        XCTAssertEqual(path.parameterCount, HeadsAllSlotPath.parameterCount)
        XCTAssertEqual(readParameterKeys(recorder, manifest: manifest, component: "heads_all_slot_path"),
                       parameterKeys(manifest, component: "heads_all_slot_path"))
    }

    /// The bundled centres hold the eight revived layers of the prediction path, 192 numbers each.
    func testBundledCentresHoldTheEightLayers() throws {
        let centres = try PredictionHeadCentres.bundled()
        XCTAssertEqual(centres.schema, PredictionHeadCentres.expectedSchema)
        XCTAssertEqual(centres.layers.count, 8)
        for (key, layer) in centres.layers {
            XCTAssertEqual(layer.centreMean.count, 192, key)
            XCTAssertTrue(layer.centreMean.allSatisfy { $0.isFinite }, key)
        }
    }

    /// One model step on synthetic weights: six finite tokens, five finite predicted tokens, latents
    /// [1, 768, 192], last step [1, 6, 192]; the context fills one model step per call.
    func testOneModelStepOnSyntheticWeightsHasTheShapes() throws {
        let manifest = try manifest()
        let model = try ForwardModel.released(from: SyntheticTensorSource(manifest: manifest),
                                              centres: try PredictionHeadCentres.bundled())
        let inputs = try syntheticInputs()
        let pass = try model.step(inputs: inputs, previousExecutedCommand: nil, at: ModelStep(count: 0))
        XCTAssertEqual(pass.tokens.count, 6)
        XCTAssertEqual(Set(pass.predictedTokens.keys), Set(SenseName.goalScored))
        XCTAssertEqual(pass.transformer.latents.shape, [1, 768, 192])
        XCTAssertEqual(pass.transformer.lastStep.shape, [1, 6, 192])
        XCTAssertEqual(pass.transformer.contextUpdate.shape, [1, 6, 192])
        XCTAssertEqual(pass.achieved.shape, [1, 26, 10])
        XCTAssertTrue(NeuralOps.isFinite(pass.transformer.latents))
        for (sense, token) in pass.tokens {
            // The shared output LayerNorm without affine parameters gives a root mean square of
            // sqrt(v / (v + 1e-5)) for a pooled vector of variance v, so at most 1. The variance
            // of the synthetic weights' pooled vectors is checked, not assumed: on the first Mac
            // build (November 4, 2026) the vision one measured 3.2e-7 (root mean square 0.176).
            let pooled = NeuralOps.floats(try XCTUnwrap(pass.pooled[sense]))
            let mean = pooled.reduce(0, +) / Float(pooled.count)
            let variance = pooled.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(pooled.count)
            let expected = (variance / (variance + 1e-5)).squareRoot()
            XCTAssertTrue(token.values.allSatisfy { $0.isFinite }, sense.rawValue)
            XCTAssertLessThanOrEqual(token.rootMeanSquare, 1.0001, sense.rawValue)
            XCTAssertEqual(token.rootMeanSquare, expected, accuracy: 1e-3,
                           "\(sense.rawValue): pooled variance \(variance)")
        }
        XCTAssertEqual(model.filledContextSteps, 1)
        // a second model step with an executed command exercises the motor-slot back-fill
        let command = Command.zero(scale: .centimetres)
        _ = try model.step(inputs: inputs, previousExecutedCommand: command, at: ModelStep(count: 1))
        XCTAssertEqual(model.filledContextSteps, 2)
        // a command at the wrong scale is refused
        XCTAssertThrowsError(try model.step(inputs: inputs, previousExecutedCommand: Command.zero(scale: .normalized),
                                            at: ModelStep(count: 2)))
        model.reset()
        XCTAssertEqual(model.filledContextSteps, 0)
    }

    /// A missing active sense is refused by name.
    func testMissingSenseIsRefused() throws {
        let model = try ForwardModel.released(from: SyntheticTensorSource(manifest: try manifest()),
                                              centres: try PredictionHeadCentres.bundled())
        var inputs = try syntheticInputs()
        inputs[.audio] = nil
        XCTAssertThrowsError(try model.step(inputs: inputs, previousExecutedCommand: nil, at: ModelStep(count: 0))) { error in
            XCTAssertEqual(error as? ModelPortError, .missingInput(.audio))
        }
    }

    /// The causal mask is 0 on and below the diagonal and -infinity above it
    /// (torch.triu(ones, diagonal=1) as a boolean attn_mask).
    func testCausalMaskRule() {
        let values = NeuralOps.floats(MultiheadAttentionLayer.causalMask(length: 4))
        for row in 0 ..< 4 {
            for column in 0 ..< 4 {
                let value = values[row * 4 + column]
                if column > row {
                    XCTAssertEqual(value, -Float.infinity)
                } else {
                    XCTAssertEqual(value, 0)
                }
            }
        }
    }

    /// The context buffer: append drops the oldest model step and puts the frame last; replaceNewest
    /// changes one token of the newest model step only.
    func testContextBufferShiftAndBackFill() {
        var buffer = ContextBuffer(contextLength: 3, senseCount: 2, width: 2)
        XCTAssertEqual(buffer.frames.shape, [1, 4, 2])
        buffer.append(MLXArray([Float](arrayLiteral: 1, 1, 2, 2), [1, 2, 2]))
        buffer.append(MLXArray([Float](arrayLiteral: 3, 3, 4, 4), [1, 2, 2]))
        XCTAssertEqual(NeuralOps.floats(buffer.frames), [1, 1, 2, 2, 3, 3, 4, 4])
        buffer.append(MLXArray([Float](arrayLiteral: 5, 5, 6, 6), [1, 2, 2]))
        XCTAssertEqual(NeuralOps.floats(buffer.frames), [3, 3, 4, 4, 5, 5, 6, 6])
        XCTAssertEqual(buffer.filledSteps, 2)
        buffer.replaceNewest(senseIndex: 1, with: MLXArray([Float](arrayLiteral: 9, 9), [1, 2]))
        XCTAssertEqual(NeuralOps.floats(buffer.frames), [3, 3, 4, 4, 5, 5, 9, 9])
        buffer.replaceNewest(senseIndex: 0, with: MLXArray([Float](arrayLiteral: 7, 7), [1, 2]))
        XCTAssertEqual(NeuralOps.floats(buffer.frames), [3, 3, 4, 4, 7, 7, 9, 9])
        buffer.reset()
        XCTAssertEqual(NeuralOps.floats(buffer.frames), [Float](repeating: 0, count: 8))
    }

    /// The Swin block keeps the sequence length when it is not a multiple of the window (1,160 = 165 x 7
    /// + 5 on the released cameras; here 10 = 7 + 3) on both the plain and the shifted block.
    func testSwinBlockKeepsLengthWithPadding() throws {
        let source = SyntheticTensorSource(manifest: try manifest())
        let encoder = try VisionEncoder(source: source.scoped("encoder_registry.encoders.vision."))
        let patches = MLXArray((0 ..< 10 * 768).map { Float(cos(Double($0))) * 0.5 }, [1, 10, 768])
        let angles = MLXArray((0 ..< 10 * 4).map { Float(sin(Double($0))) }, [1, 10, 4])
        let features = encoder.backbone(patches: patches, angles: angles)
        XCTAssertEqual(features.shape, [1, 10, 192])
        XCTAssertEqual(encoder.head(features).shape, [1, 1, 192])
        XCTAssertTrue(NeuralOps.isFinite(features))
    }

    // MARK: With the weights and the WI-34 fixtures

    private func releasedWeightsURL(manifest: WeightManifest) -> URL {
        if let override = ProcessInfo.processInfo.environment["UEMA_WEIGHTS_PATH"], !override.isEmpty {
            return URL(fileURLWithPath: override)
        }
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0 ..< 5 {
            root = root.deletingLastPathComponent()
        }
        return root.appendingPathComponent(manifest.source.path)
    }

    private func fixture(_ state: String) throws -> [String: MLXArray] {
        guard let url = Bundle.module.url(forResource: state, withExtension: "safetensors",
                                          subdirectory: "Fixtures/model_pass") else {
            throw XCTSkip("Fixtures/model_pass/\(state).safetensors is not in the test bundle")
        }
        return try loadArrays(url: url)
    }

    /// The fixture tensor `key` without its leading batch extent of 1, as a UEMACore array.
    private func shaped(_ arrays: [String: MLXArray], _ key: String) throws -> ShapedArray {
        let array = try XCTUnwrap(arrays[key], "the fixture has no \(key)")
        var shape = array.shape
        if shape.first == 1 && shape.count > 1 {
            shape.removeFirst()
        }
        return try ShapedArray(shape: shape, values: NeuralOps.floats(array))
    }

    private func fixtureInputs(_ arrays: [String: MLXArray]) throws -> [SenseName: EncoderInput] {
        [
            .vision: EncoderInput(sense: .vision, primary: try shaped(arrays, "input/vision/patches"),
                                  auxiliary: try shaped(arrays, "input/vision/angular_pos")),
            .tactile: EncoderInput(sense: .tactile, primary: try shaped(arrays, "input/tactile")),
            .audio: EncoderInput(sense: .audio, primary: try shaped(arrays, "input/audio/spectral"),
                                 auxiliary: try shaped(arrays, "input/audio/spatial")),
            .proprioception: EncoderInput(sense: .proprioception, primary: try shaped(arrays, "input/proprioception")),
            .motor: EncoderInput(sense: .motor, primary: try shaped(arrays, "input/motor")),
            .vestibular: EncoderInput(sense: .vestibular, primary: try shaped(arrays, "input/vestibular")),
        ]
    }

    /// The acceptance of WI-35: on each WI-34 state, the first model step after a reset reproduces the
    /// fixture's six tokens and five predicted tokens within ModelPassTolerance (1e-3 token units), no
    /// NaN. Prints every measured difference for section 9.
    func testFixturePassesMatchWithinTheDeclaredTolerance() throws {
        let manifest = try manifest()
        let url = releasedWeightsURL(manifest: manifest)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("model.safetensors is not at \(url.path); run fetch_weights.py --accept-license or set UEMA_WEIGHTS_PATH")
        }
        let weights = try SafetensorsLoader(manifest: manifest).load(url: url)
        let centres = try PredictionHeadCentres.bundled()
        let line = ModelPassTolerance.maxAbsoluteDifference
        for state in ["step_000492800", "step_000493674", "step_000493980"] {
            let arrays = try fixture(state)
            let bias = try XCTUnwrap(arrays["forward/memory_bias"], "\(state) has no forward/memory_bias")
            let model = try ForwardModel.released(from: weights, centres: centres,
                                                  memoryBias: FixedMemoryBias(bias: bias))
            let pass = try model.step(inputs: try fixtureInputs(arrays), previousExecutedCommand: nil,
                                      at: ModelStep(count: 0))
            for sense in SenseName.active {
                let pooled = try XCTUnwrap(pass.pooled[sense])
                let fixturePooled = try XCTUnwrap(arrays["encoder_output/\(sense.rawValue)"])
                let token = MLXArray(try XCTUnwrap(pass.tokens[sense]).values, [192])
                let fixtureToken = try XCTUnwrap(arrays["token/\(sense.rawValue)"])
                let pooledDifference = maxAbsoluteDifference(pooled, fixturePooled)
                let tokenDifference = maxAbsoluteDifference(token, fixtureToken)
                print("WI-35 \(state) \(sense.rawValue): pooled vector largest difference \(pooledDifference) "
                      + "(largest fixture magnitude \(NeuralOps.floats(fixturePooled).map { abs($0) }.max() ?? 0)); "
                      + "token largest difference \(tokenDifference) against the line \(line)")
                XCTAssertLessThanOrEqual(tokenDifference, line, "\(state) token \(sense.rawValue)")
            }
            for sense in SenseName.goalScored {
                let predicted = MLXArray(try XCTUnwrap(pass.predictedTokens[sense]).values, [192])
                let fixturePredicted = try XCTUnwrap(arrays["predicted/\(sense.rawValue)"])
                let difference = maxAbsoluteDifference(predicted, fixturePredicted)
                print("WI-35 \(state) \(sense.rawValue): predicted token largest difference \(difference) against the line \(line)")
                XCTAssertLessThanOrEqual(difference, line, "\(state) predicted \(sense.rawValue)")
            }
            let latents = try XCTUnwrap(arrays["forward/latents"])
            print("WI-35 \(state): latents largest difference \(maxAbsoluteDifference(pass.transformer.latents, latents)) "
                  + "(the memory bias reaches magnitude \(NeuralOps.floats(bias).map { abs($0) }.max() ?? 0))")
            XCTAssertTrue(NeuralOps.isFinite(pass.transformer.latents), "\(state) latents")
        }
    }
}
