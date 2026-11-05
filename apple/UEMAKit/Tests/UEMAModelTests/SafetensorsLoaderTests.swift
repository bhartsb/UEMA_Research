// SafetensorsLoaderTests.swift: the tests of Sources/UEMAModel/SafetensorsLoader.swift (work
// item WI-31). Nothing in this file was compiled on the machine that wrote it (no Swift
// toolchain on Windows).
//
// The loader tests run on the bundled manifest alone, through a reader that builds
// zero-filled arrays of the manifest's shapes and dtypes (MLXArray.zeros(_:dtype:stream:),
// Source/MLX/Factory.swift of mlx-swift, read November 4, 2026: "static public func zeros(
// _ shape: some Collection<Int>, dtype: DType, stream: StreamOrDevice = .default) -> MLXArray";
// stream .cpu because Metal computes no float64 and the manifest has 49 F64 buffers). Two tests
// build small safetensors files (an 8-byte little-endian header length, the JSON header, the
// data section) and read them with SafetensorsFileReader, the loader's default reader;
// MLXArray.asArray(_:) returns an array's values as a Swift array. The last test reads the
// released model.safetensors (73,119,988 bytes, not in git) and skips when the file is absent.
//
// The Linear layer used below: Source/MLXNN/Linear.swift, "public init(_ inputDimensions: Int,
// _ outputDimensions: Int, bias: Bool = true)", "@ParameterInfo public var weight: MLXArray",
// "@ParameterInfo public var bias: MLXArray?"; its parameter keys are therefore "weight" and
// "bias", the PyTorch names, and its weight shape is [outputDimensions, inputDimensions], the
// PyTorch layout, so wellness.grounded.tactile_comfort.net.0.weight [32, 8] fits Linear(8, 32).
// NestedItem cases (Source/MLX/Nested.swift): none, value(Element), array([NestedItem]),
// dictionary([Key: NestedItem]); NestedDictionary subscript(key:) returns NestedItem?.

import XCTest
import Foundation
import MLX
import MLXNN
import UEMACore
@testable import UEMAModel

/// A reader that builds every manifest tensor as zeros of its shape and dtype, with optional
/// removed keys, added keys and replaced arrays, so the loader's checks run without the file.
private struct ManifestZerosReader: SafetensorsReading {
    let manifest: WeightManifest
    var removedKeys: Set<String> = []
    var addedArrays: [String: MLXArray] = [:]
    var replacedArrays: [String: MLXArray] = [:]

    func readArrays(url: URL) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for tensor in manifest.tensors where !removedKeys.contains(tensor.key) {
            if let replacement = replacedArrays[tensor.key] {
                out[tensor.key] = replacement
            } else {
                out[tensor.key] = MLXArray.zeros(tensor.shape, dtype: tensor.dtype.mlxDType, stream: .cpu)
            }
        }
        for (key, array) in addedArrays {
            out[key] = array
        }
        return out
    }
}

final class SafetensorsLoaderTests: XCTestCase {
    /// The URL the fake reader is handed; it never opens it.
    private let fakeURL = URL(fileURLWithPath: "/nonexistent/model.safetensors")

    /// The key of one forward-model parameter used by the refusal tests.
    private let finalNormWeight = "forward_model.final_norm.weight"

    private func bundledManifest() throws -> WeightManifest {
        try WeightManifest.bundled(in: UEMAModelModule.resourceBundle)
    }

    // MARK: Manifest alone

    /// The bundled manifest decodes and carries the counts dump_weight_manifest.py recorded for
    /// the released file: 802 tensors, 676 parameters of 18,131,745 elements, 126 buffers,
    /// 9,926,358 parameter elements in the acting path, model step 494,025.
    func testBundledManifestDecodesWithTheRecordedCounts() throws {
        let manifest = try bundledManifest()
        XCTAssertEqual(manifest.schema, WeightManifest.expectedSchema)
        XCTAssertEqual(manifest.tensors.count, 802)
        XCTAssertEqual(manifest.counts.tensors, 802)
        XCTAssertEqual(manifest.counts.parameterTensors, 676)
        XCTAssertEqual(manifest.counts.bufferTensors, 126)
        XCTAssertEqual(manifest.counts.parameterElements, 18_131_745)
        XCTAssertEqual(manifest.counts.bufferElements, 168_674)
        XCTAssertEqual(manifest.counts.inferenceSetParameterElements, 9_926_358)
        XCTAssertEqual(manifest.parameterElements, 18_131_745)
        XCTAssertEqual(manifest.source.bytes, 73_119_988)
        XCTAssertEqual(manifest.source.modelStep, 494_025)
        XCTAssertEqual(manifest.source.receiptTotalParams, 18_131_745)
        XCTAssertEqual(manifest.components.count, 24)
    }

    /// The loader's expected-key set equals the manifest's key set (the acceptance line "with
    /// the manifest alone ... the expected-key set equals the manifest's"), and a file holding
    /// exactly those keys loads with 0 missing and 0 shape mismatches.
    func testExpectedKeySetEqualsTheManifestKeys() throws {
        let manifest = try bundledManifest()
        let manifestKeys = Set(manifest.tensors.map { $0.key })
        XCTAssertEqual(manifestKeys.count, 802, "the manifest holds 802 distinct keys")
        let loader = SafetensorsLoader(manifest: manifest, reader: ManifestZerosReader(manifest: manifest))
        XCTAssertEqual(loader.expectedKeys, manifestKeys)

        let loaded = try loader.load(url: fakeURL)
        XCTAssertEqual(Set(loaded.arrays.keys), manifestKeys)
        XCTAssertEqual(loaded.parameterElements, 18_131_745, "zeros of the manifest shapes hold the receipt's count")
        XCTAssertEqual(loaded.totalElements, 18_131_745 + 168_674)
        let report = loaded.report
        XCTAssertEqual(report.missing, 0)
        XCTAssertEqual(report.shapeMismatches, 0)
        XCTAssertEqual(report.lines.count, 802)
        XCTAssertEqual(report.loadedTensors, 334, "the acting path's ten components hold 334 tensors")
        XCTAssertEqual(report.skippedTensors, 468)
        XCTAssertEqual(report.loadedParameterElements, 9_926_358)
        XCTAssertEqual(report.skippedParameterElements, 18_131_745 - 9_926_358)
    }

    /// Each tensor's component equals the split dump_weight_manifest.py applied to its key, the
    /// per-component totals equal the components block, and the manifest's inference-set
    /// components are exactly the ten WeightComponent cases, each key carrying its prefix.
    func testComponentsAgreeWithTheTensorRecords() throws {
        let manifest = try bundledManifest()
        var parameterElements: [String: Int] = [:]
        var bufferElements: [String: Int] = [:]
        var tensorCounts: [String: Int] = [:]
        for tensor in manifest.tensors {
            XCTAssertEqual(WeightManifest.component(ofKey: tensor.key), tensor.component, tensor.key)
            XCTAssertEqual(tensor.elements, tensor.shapeElements, tensor.key)
            tensorCounts[tensor.component, default: 0] += 1
            if tensor.kind == .parameter {
                parameterElements[tensor.component, default: 0] += tensor.elements
            } else {
                bufferElements[tensor.component, default: 0] += tensor.elements
            }
            let inActingPath = WeightComponent(rawValue: tensor.component) != nil
            XCTAssertEqual(tensor.inferenceSet, inActingPath, tensor.key)
        }
        for (name, block) in manifest.components {
            XCTAssertEqual(block.parameterElements, parameterElements[name] ?? 0, name)
            XCTAssertEqual(block.bufferElements, bufferElements[name] ?? 0, name)
            XCTAssertEqual(block.tensors, tensorCounts[name] ?? 0, name)
            XCTAssertEqual(block.parameterElements, block.receiptComponentParams, name)
            XCTAssertEqual(block.inferenceSet, WeightComponent(rawValue: name) != nil, name)
        }
        let inferenceSet = Set(manifest.components.filter { $0.value.inferenceSet }.map { $0.key })
        XCTAssertEqual(inferenceSet, Set(WeightComponent.allCases.map { $0.rawValue }))
        for component in WeightComponent.allCases {
            let records = manifest.tensors(inComponent: component.rawValue)
            XCTAssertFalse(records.isEmpty, component.rawValue)
            for tensor in records {
                XCTAssertTrue(tensor.key.hasPrefix(component.keyPrefix), "\(tensor.key) under \(component.keyPrefix)")
            }
        }
        XCTAssertEqual(manifest.tensors(inComponent: "forward_model").count, 104)
        XCTAssertEqual(manifest.tensors(inComponent: "enc_vision").count, 42)
    }

    /// The element type table agrees with the safetensors format on bytes per element and maps
    /// the four dtypes of the released file onto MLX's.
    func testDTypeTable() {
        XCTAssertEqual(SafetensorsDType.f32.bytesPerElement, 4)
        XCTAssertEqual(SafetensorsDType.f64.bytesPerElement, 8)
        XCTAssertEqual(SafetensorsDType.i64.bytesPerElement, 8)
        XCTAssertEqual(SafetensorsDType.bool.bytesPerElement, 1)
        XCTAssertEqual(SafetensorsDType.f32.mlxDType, .float32)
        XCTAssertEqual(SafetensorsDType.f64.mlxDType, .float64)
        XCTAssertEqual(SafetensorsDType.i64.mlxDType, .int64)
        XCTAssertEqual(SafetensorsDType.bool.mlxDType, .bool)
        XCTAssertEqual(SafetensorsDType(rawValue: "F32"), .f32)
        XCTAssertNil(SafetensorsDType(rawValue: "F8"))
    }

    /// A manifest whose schema field is another name is refused with both names.
    func testOtherSchemaIsRefused() throws {
        let json = """
        {"schema": "other_schema", "generated_at": "", "source": {"path": "", "bytes": 0, "sha256": "",
         "header_bytes": 0, "model_step": 0, "receipt_generated_at": "", "receipt_total_params": 0},
         "counts": {"tensors": 0, "parameter_tensors": 0, "buffer_tensors": 0, "parameter_elements": 0,
         "buffer_elements": 0, "inference_set_parameter_elements": 0, "dtypes": {}},
         "components": {}, "tensors": []}
        """
        XCTAssertThrowsError(try WeightManifest(data: Data(json.utf8))) { error in
            guard case SafetensorsLoaderError.manifestSchema(let found, let expected) = error else {
                return XCTFail("expected manifestSchema, got \(error)")
            }
            XCTAssertEqual(found, "other_schema")
            XCTAssertEqual(expected, WeightManifest.expectedSchema)
        }
    }

    /// A record whose elements field disagrees with its shape is refused naming the key.
    func testInconsistentRecordIsRefused() throws {
        let json = """
        {"schema": "uema_weight_manifest_v1", "generated_at": "", "source": {"path": "", "bytes": 0,
         "sha256": "", "header_bytes": 0, "model_step": 0, "receipt_generated_at": "",
         "receipt_total_params": 0},
         "counts": {"tensors": 1, "parameter_tensors": 1, "buffer_tensors": 0, "parameter_elements": 6,
         "buffer_elements": 0, "inference_set_parameter_elements": 0, "dtypes": {"F32": 1}},
         "components": {}, "tensors": [{"key": "a.b", "shape": [2, 3], "dtype": "F32", "elements": 7,
         "bytes": 28, "component": "a", "kind": "parameter", "inference_set": false}]}
        """
        XCTAssertThrowsError(try WeightManifest(data: Data(json.utf8))) { error in
            guard case SafetensorsLoaderError.manifestInconsistent(let problems) = error else {
                return XCTFail("expected manifestInconsistent, got \(error)")
            }
            XCTAssertTrue(problems.contains { $0.contains("a.b") }, "\(problems)")
        }
    }

    /// A file lacking one manifest key is refused and the key is named.
    func testMissingKeyIsNamed() throws {
        let manifest = try bundledManifest()
        var reader = ManifestZerosReader(manifest: manifest)
        reader.removedKeys = [finalNormWeight]
        let loader = SafetensorsLoader(manifest: manifest, reader: reader)
        XCTAssertThrowsError(try loader.load(url: fakeURL)) { error in
            guard case SafetensorsLoaderError.keysDiffer(let missing, let extra) = error else {
                return XCTFail("expected keysDiffer, got \(error)")
            }
            XCTAssertEqual(missing, [finalNormWeight])
            XCTAssertEqual(extra, [])
            XCTAssertTrue(String(describing: error).contains(finalNormWeight))
        }
    }

    /// A file holding one key the manifest lacks is refused and the key is named.
    func testExtraKeyIsNamed() throws {
        let manifest = try bundledManifest()
        var reader = ManifestZerosReader(manifest: manifest)
        let extraKey = "forward_model.unlisted.weight"
        reader.addedArrays = [extraKey: MLXArray.zeros([3], dtype: .float32, stream: .cpu)]
        let loader = SafetensorsLoader(manifest: manifest, reader: reader)
        XCTAssertThrowsError(try loader.load(url: fakeURL)) { error in
            guard case SafetensorsLoaderError.keysDiffer(let missing, let extra) = error else {
                return XCTFail("expected keysDiffer, got \(error)")
            }
            XCTAssertEqual(missing, [])
            XCTAssertEqual(extra, [extraKey])
        }
    }

    /// A tensor of another shape is refused with the key and both shapes.
    func testShapeMismatchIsNamed() throws {
        let manifest = try bundledManifest()
        var reader = ManifestZerosReader(manifest: manifest)
        reader.replacedArrays = [finalNormWeight: MLXArray.zeros([191], dtype: .float32, stream: .cpu)]
        let loader = SafetensorsLoader(manifest: manifest, reader: reader)
        XCTAssertThrowsError(try loader.load(url: fakeURL)) { error in
            guard case SafetensorsLoaderError.tensorsDiffer(let mismatches) = error else {
                return XCTFail("expected tensorsDiffer, got \(error)")
            }
            XCTAssertEqual(mismatches.count, 1)
            XCTAssertEqual(mismatches.first?.key, finalNormWeight)
            XCTAssertEqual(mismatches.first?.manifestShape, [192])
            XCTAssertEqual(mismatches.first?.fileShape, [191])
            XCTAssertEqual(mismatches.first?.shapeDiffers, true)
            XCTAssertEqual(mismatches.first?.dtypeDiffers, false)
        }
    }

    /// A tensor of another element type is refused with the key and both types.
    func testDTypeMismatchIsNamed() throws {
        let manifest = try bundledManifest()
        var reader = ManifestZerosReader(manifest: manifest)
        reader.replacedArrays = [finalNormWeight: MLXArray.zeros([192], dtype: .float16, stream: .cpu)]
        let loader = SafetensorsLoader(manifest: manifest, reader: reader)
        XCTAssertThrowsError(try loader.load(url: fakeURL)) { error in
            guard case SafetensorsLoaderError.tensorsDiffer(let mismatches) = error else {
                return XCTFail("expected tensorsDiffer, got \(error)")
            }
            XCTAssertEqual(mismatches.count, 1)
            XCTAssertEqual(mismatches.first?.key, finalNormWeight)
            XCTAssertEqual(mismatches.first?.shapeDiffers, false)
            XCTAssertEqual(mismatches.first?.dtypeDiffers, true)
            XCTAssertEqual(mismatches.first?.fileDType, .float16)
            XCTAssertEqual(mismatches.first?.manifestDType, .f32)
        }
    }

    /// parameters(for:) strips the component prefix and nests the keys: the forward model's
    /// "blocks" becomes an array of 4 and "final_norm.weight" a [192] value; buffers stay out
    /// unless asked for.
    func testParametersNestByPrefix() throws {
        let manifest = try bundledManifest()
        let loader = SafetensorsLoader(manifest: manifest, reader: ManifestZerosReader(manifest: manifest))
        let loaded = try loader.load(url: fakeURL)

        let flat = loaded.flatParameters(keyPrefix: WeightComponent.forwardModel.keyPrefix)
        XCTAssertEqual(flat.count, 100, "104 forward_model tensors minus 4 causal_mask buffers")
        XCTAssertNotNil(flat["final_norm.weight"])
        XCTAssertNil(flat["blocks.0.causal_mask"])
        let withBuffers = loaded.flatParameters(keyPrefix: WeightComponent.forwardModel.keyPrefix, includeBuffers: true)
        XCTAssertEqual(withBuffers.count, 104)
        XCTAssertEqual(withBuffers["blocks.0.causal_mask"]?.shape, [128, 128])
        XCTAssertEqual(loaded.buffers(for: .forwardModel).count, 4)
        XCTAssertEqual(loaded.buffers(for: .predictionHeads).count, 1)
        XCTAssertEqual(loaded.buffers(for: .wellness).count, 0)

        let nested = loaded.parameters(for: .forwardModel)
        guard case .array(let blocks)? = nested["blocks"] else {
            return XCTFail("blocks is not an array level")
        }
        XCTAssertEqual(blocks.count, 4)
        guard case .dictionary(let finalNorm)? = nested["final_norm"] else {
            return XCTFail("final_norm is not a dictionary level")
        }
        guard case .value(let weight)? = finalNorm["weight"] else {
            return XCTFail("final_norm.weight is not a value")
        }
        XCTAssertEqual(weight.shape, [192])

        let vision = loaded.flatParameters(keyPrefix: WeightComponent.visionEncoder.keyPrefix)
        XCTAssertEqual(vision.count, 42)
        for key in vision.keys {
            XCTAssertFalse(key.hasPrefix("encoder_registry"), key)
        }
    }

    /// The unflattened parameters of one wellness channel's first layer fill an MLXNN Linear
    /// through Module.update(parameters:verify: .all): weight [32, 8] and bias [32] land on the
    /// layer's own properties.
    func testParametersFeedALinearLayer() throws {
        let manifest = try bundledManifest()
        let loader = SafetensorsLoader(manifest: manifest, reader: ManifestZerosReader(manifest: manifest))
        let loaded = try loader.load(url: fakeURL)
        let prefix = "wellness.grounded.tactile_comfort.net.0."
        let flat = loaded.flatParameters(keyPrefix: prefix)
        XCTAssertEqual(Set(flat.keys), ["weight", "bias"])
        XCTAssertEqual(flat["weight"]?.shape, [32, 8])
        XCTAssertEqual(flat["bias"]?.shape, [32])

        let layer = Linear(8, 32)
        _ = try layer.update(parameters: loaded.parameters(keyPrefix: prefix), verify: .all)
        XCTAssertEqual(layer.weight.shape, [32, 8])
        XCTAssertEqual(layer.bias?.shape, [32])

        let wrongLayer = Linear(9, 32)
        XCTAssertThrowsError(try wrongLayer.update(parameters: loaded.parameters(keyPrefix: prefix), verify: .all),
                             "a [32, 8] weight must not fit a Linear(9, 32)")
    }

    /// Through UEMACore's WeightsLoading (Protocols.swift, WI-20) the loader returns the checked
    /// arrays and a WeightsReport of 334 loaded keys, 468 skipped keys and 9,991,895 loaded
    /// elements (9,926,358 parameter elements + 65,537 buffer elements, manifest records), and a
    /// missing key throws naming it.
    func testWeightsLoadingConformanceReportsLoadedAndSkippedKeys() throws {
        let manifest = try bundledManifest()
        let loader = SafetensorsLoader(manifest: manifest, reader: ManifestZerosReader(manifest: manifest))
        let contract: any WeightsLoading<LoadedWeights> = loader
        let result = try contract.load(from: fakeURL)
        let report: WeightsReport = result.report
        XCTAssertEqual(report.fileURL, fakeURL)
        XCTAssertEqual(report.loadedKeys.count, 334)
        XCTAssertEqual(report.skippedKeys.count, 468)
        XCTAssertEqual(Set(report.loadedKeys).union(report.skippedKeys), Set(manifest.tensors.map { $0.key }))
        XCTAssertTrue(Set(report.loadedKeys).isDisjoint(with: report.skippedKeys))
        XCTAssertEqual(report.loadedElements, 9_991_895)
        XCTAssertEqual(result.weights.arrays.count, 802)
        XCTAssertTrue(report.loadedKeys.contains(finalNormWeight))
        XCTAssertTrue(report.skippedKeys.contains("contact_head.ctx2.bias"))

        let entry = try XCTUnwrap(manifest.byKey[finalNormWeight]).manifestEntry
        XCTAssertEqual(entry.shape, [192])
        XCTAssertEqual(entry.dtype, "F32")
        XCTAssertEqual(entry.kind, "parameter")
        XCTAssertTrue(entry.inferenceSet)

        var reader = ManifestZerosReader(manifest: manifest)
        reader.removedKeys = [finalNormWeight]
        let refusing: any WeightsLoading<LoadedWeights> = SafetensorsLoader(manifest: manifest, reader: reader)
        XCTAssertThrowsError(try refusing.load(from: fakeURL)) { error in
            XCTAssertTrue(String(describing: error).contains(finalNormWeight), "\(error)")
        }
    }

    // MARK: The file reader on small files

    /// The bytes of a safetensors file: the header's length as a little-endian UInt64, the
    /// header JSON text, then the data section.
    private func safetensorsBytes(header: String, data: Data) -> Data {
        let json = Data(header.utf8)
        let length = UInt64(json.count)
        var out = Data((0..<8).map { UInt8(truncatingIfNeeded: length >> UInt64(8 * $0)) })
        out.append(json)
        out.append(data)
        return out
    }

    /// A file of one F32 [2, 2] tensor, one F64 scalar and one BOOL [3] tensor, with a
    /// __metadata__ entry, reads back with each key's shape, dtype and values.
    func testFileReaderReadsF32F64AndBool() throws {
        let floats: [Float] = [1, 2, 3, 4]
        let doubles: [Double] = [2.5]
        var section = floats.withUnsafeBufferPointer { Data(buffer: $0) }
        section.append(doubles.withUnsafeBufferPointer { Data(buffer: $0) })
        section.append(contentsOf: [1, 0, 1] as [UInt8])
        XCTAssertEqual(section.count, 27, "16 + 8 + 3 bytes")
        let header = """
        {"__metadata__": {"format": "pt"},
         "a.w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
         "b.s": {"dtype": "F64", "shape": [], "data_offsets": [16, 24]},
         "c.m": {"dtype": "BOOL", "shape": [3], "data_offsets": [24, 27]}}
        """
        let bytes = safetensorsBytes(header: header, data: section)

        let parsed = try SafetensorsFileReader.parseHeader(bytes)
        XCTAssertEqual(parsed.headerBytes, 8 + Data(header.utf8).count)
        XCTAssertEqual(parsed.entries.map { $0.key }, ["a.w", "b.s", "c.m"])
        XCTAssertEqual(parsed.entries.map { $0.dtype }, [.f32, .f64, .bool])
        XCTAssertEqual(parsed.entries.map { $0.dataRange }, [0..<16, 16..<24, 24..<27])

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("SafetensorsLoaderTests-\(UUID().uuidString).safetensors")
        try bytes.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let arrays = try SafetensorsFileReader().readArrays(url: url)
        XCTAssertEqual(Set(arrays.keys), ["a.w", "b.s", "c.m"])
        let weight = try XCTUnwrap(arrays["a.w"])
        XCTAssertEqual(weight.shape, [2, 2])
        XCTAssertEqual(weight.dtype, .float32)
        XCTAssertEqual(weight.asArray(Float.self), floats)
        let scalar = try XCTUnwrap(arrays["b.s"])
        XCTAssertEqual(scalar.shape, [])
        XCTAssertEqual(scalar.dtype, .float64)
        let mask = try XCTUnwrap(arrays["c.m"])
        XCTAssertEqual(mask.shape, [3])
        XCTAssertEqual(mask.dtype, .bool)
        XCTAssertEqual(mask.asArray(Bool.self), [true, false, true])
        XCTAssertEqual(try SafetensorsFileReader().readHeader(url: url), parsed)
    }

    /// Short files, a header that is not a JSON object, an entry without a shape, an unknown
    /// dtype, a byte range that disagrees with shape x dtype and a range past the end of the
    /// data section are each refused, naming the key where there is one.
    func testFileReaderRefusesBadFiles() {
        XCTAssertThrowsError(try SafetensorsFileReader.parseHeader(Data([1, 2, 3, 4]))) { error in
            guard case SafetensorsFileError.truncatedHeader = error else {
                return XCTFail("expected truncatedHeader, got \(error)")
            }
        }
        var tooLong = Data([255, 0, 0, 0, 0, 0, 0, 0])
        tooLong.append(Data("{}".utf8))
        XCTAssertThrowsError(try SafetensorsFileReader.parseHeader(tooLong)) { error in
            guard case SafetensorsFileError.truncatedHeader(let fileBytes, _) = error else {
                return XCTFail("expected truncatedHeader, got \(error)")
            }
            XCTAssertEqual(fileBytes, 10)
        }
        XCTAssertThrowsError(try SafetensorsFileReader.parseHeader(safetensorsBytes(header: "[1, 2]", data: Data()))) { error in
            guard case SafetensorsFileError.headerNotJSON = error else {
                return XCTFail("expected headerNotJSON, got \(error)")
            }
        }
        let noShape = #"{"k.a": {"dtype": "F32", "data_offsets": [0, 4]}}"#
        XCTAssertThrowsError(try SafetensorsFileReader.parseHeader(safetensorsBytes(header: noShape, data: Data(count: 4)))) { error in
            guard case SafetensorsFileError.malformedEntry(let key) = error else {
                return XCTFail("expected malformedEntry, got \(error)")
            }
            XCTAssertEqual(key, "k.a")
        }
        let unknown = #"{"k.b": {"dtype": "F8_E4M3", "shape": [4], "data_offsets": [0, 4]}}"#
        XCTAssertThrowsError(try SafetensorsFileReader.parseHeader(safetensorsBytes(header: unknown, data: Data(count: 4)))) { error in
            guard case SafetensorsFileError.unknownDType(let key, let dtype) = error else {
                return XCTFail("expected unknownDType, got \(error)")
            }
            XCTAssertEqual(key, "k.b")
            XCTAssertEqual(dtype, "F8_E4M3")
        }
        let wrongCount = #"{"k.c": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 12]}}"#
        XCTAssertThrowsError(try SafetensorsFileReader.parseHeader(safetensorsBytes(header: wrongCount, data: Data(count: 16)))) { error in
            guard case SafetensorsFileError.badByteRange(let key, _, _, let expectedBytes, _) = error else {
                return XCTFail("expected badByteRange, got \(error)")
            }
            XCTAssertEqual(key, "k.c")
            XCTAssertEqual(expectedBytes, 16)
        }
        let pastEnd = #"{"k.d": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}"#
        XCTAssertThrowsError(try SafetensorsFileReader.parseHeader(safetensorsBytes(header: pastEnd, data: Data(count: 8)))) { error in
            guard case SafetensorsFileError.badByteRange(let key, _, let end, _, let dataBytes) = error else {
                return XCTFail("expected badByteRange, got \(error)")
            }
            XCTAssertEqual(key, "k.d")
            XCTAssertEqual(end, 16)
            XCTAssertEqual(dataBytes, 8)
        }
    }

    // MARK: With the file

    /// Where the released file is on this machine: the environment variable UEMA_WEIGHTS_PATH,
    /// else the manifest's source.path under the repository root (five path components above
    /// this file: UEMAModelTests, Tests, UEMAKit, apple, and the file name itself).
    private func releasedWeightsURL(manifest: WeightManifest) -> URL {
        if let override = ProcessInfo.processInfo.environment["UEMA_WEIGHTS_PATH"], !override.isEmpty {
            return URL(fileURLWithPath: override)
        }
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 {
            root = root.deletingLastPathComponent()
        }
        return root.appendingPathComponent(manifest.source.path)
    }

    /// With the released file: every one of the 802 keys loads, 0 missing, 0 shape mismatches,
    /// and the parameter tensors hold 18,131,745 elements (the acceptance line "with the file,
    /// 18,131,745 elements"). Skips when the file is absent.
    func testRealFileHasTheReceiptParameterCount() throws {
        let manifest = try bundledManifest()
        let url = releasedWeightsURL(manifest: manifest)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("model.safetensors is not at \(url.path); run fetch_weights.py --accept-license or set UEMA_WEIGHTS_PATH")
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        XCTAssertEqual((attributes[.size] as? NSNumber)?.intValue, manifest.source.bytes)

        let header = try SafetensorsFileReader().readHeader(url: url)
        XCTAssertEqual(header.headerBytes, manifest.source.headerBytes, "91,104 bytes, manifest source.header_bytes")
        XCTAssertEqual(header.entries.count, 802)
        for entry in header.entries {
            XCTAssertEqual(manifest.byKey[entry.key]?.shape, entry.shape, entry.key)
            XCTAssertEqual(manifest.byKey[entry.key]?.dtype, entry.dtype, entry.key)
        }

        let loader = try SafetensorsLoader.bundled(in: UEMAModelModule.resourceBundle)
        let loaded = try loader.load(url: url)
        XCTAssertEqual(loaded.arrays.count, 802)
        XCTAssertEqual(loaded.parameterElements, 18_131_745)
        XCTAssertEqual(loaded.totalElements, 18_300_419)
        let report = loaded.report
        XCTAssertEqual(report.missing, 0)
        XCTAssertEqual(report.shapeMismatches, 0)
        XCTAssertEqual(report.loadedParameterElements, 9_926_358)
        XCTAssertEqual(loaded.weightsReport.loadedElements, 9_991_895)
        print("SafetensorsLoader real file: \(loaded.parameterElements) parameter elements, \(report.loadedTensors) loaded tensors, \(report.skippedTensors) skipped, \(report.missing) missing, \(report.shapeMismatches) shape mismatches")

        let layer = Linear(8, 32)
        _ = try layer.update(parameters: loaded.parameters(keyPrefix: "wellness.grounded.tactile_comfort.net.0."),
                             verify: .all)
        XCTAssertEqual(layer.weight.shape, [32, 8])
    }
}
