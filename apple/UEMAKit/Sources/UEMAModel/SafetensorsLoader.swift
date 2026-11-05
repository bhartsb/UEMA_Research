// SafetensorsLoader.swift: the weights loader of the UEMAModel module (work item WI-31 of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-11 of
// docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html). Nothing in this file was compiled on
// the machine that wrote it (Windows 11, no Swift toolchain, no Xcode: plan section 3, fact 1).
//
// WHAT THIS FILE DOES.
// INPUT: model.safetensors (73,119,988 bytes, the checkpoint of model step 494,025; not in git,
//   fetched by fetch_weights.py; RESUME.md:9 of weights/b1_continue_B1K_skip13_492800) and
//   Resources/weight_manifest.json (written by apple/scripts/dump_weight_manifest.py, WI-12:
//   802 tensor records, each with key, shape, dtype, elements, bytes, component, kind and
//   inference_set; 676 parameter tensors of 18,131,745 elements and 126 buffer tensors of
//   168,674 elements; counts block of the manifest).
// OPERATION: SafetensorsFileReader reads the file into a dictionary from tensor name to MLXArray:
//   it parses the 8-byte header length and the JSON header with Foundation and builds each
//   array from its bytes with MLXArray(_:_:dtype:). The plan row WI-31 names MLX.loadArrays(url:)
//   for this step; MLX's safetensors parser throws "[safetensor] unsupported dtype" on F64, and
//   the released file holds 49 F64 buffers (manifest counts.dtypes; see MLXSafetensorsReader,
//   which wraps loadArrays and stays for files without F64). The dictionary's key set is
//   compared with the manifest's key set (a key in the manifest and not in the file is MISSING; a key in the file and not in the manifest is EXTRA; either
//   is an error that names every such key). Then every array's shape (MLXArray.shape, [Int])
//   and element type (MLXArray.dtype) are compared with the manifest record's shape and dtype;
//   any difference is an error naming the key and both values. The arrays are evaluated once
//   the checks pass (arrays built from bytes are already evaluated; a reader that returns lazy
//   arrays, such as MLXSafetensorsReader, has them read from disk here).
// OUTPUT: a LoadedWeights value holding the checked dictionary; its parameters(for:) gives one
//   component's arrays with the component's key prefix removed, nested through
//   ModuleParameters.unflattened so that Module.update(parameters:verify:) of MLXNN fills a
//   module tree whose property names equal the PyTorch names (WI-35 and WI-36 build those
//   trees). Its report lists every manifest tensor as loaded (component in the acting path:
//   the six encoders, forward_model, prediction_heads, motor_controller, wellness) or skipped
//   (the 14 training-only components), by name, with the counts of missing keys and shape
//   mismatches, which read 0 after a successful load (REQ-B-11's pass line).
//
// KEY LAYOUT (the Python truth). Source/Python/lem/training/checkpoint.py:376-384 writes every
// tensor under "<component>.<state_dict key>" ("flat[f\"{component}.{k}\"] = v" at line 383,
// then save_file at 384); checkpoint.py:768-775 splits each key at its first "." back into a
// component and a state_dict key on load. The encoders live under one component,
// encoder_registry, whose state_dict keys begin "encoders.<sense>.", so a vision encoder tensor
// is "encoder_registry.encoders.vision.<name>"; Source/Python/mimo_bridge/lem_model.py:1398-1414
// counts those as "enc_<sense>" when it writes the receipt's component_params, and
// apple/scripts/dump_weight_manifest.py component_of() applies the same split to the manifest's
// component field. WeightComponent.keyPrefix below is that split in Swift.
//
// MLX SWIFT SYMBOLS USED, with the documentation page of each (the pages at
// swiftpackageindex.com were fetched November 4, 2026 and returned only their navigation to the
// fetch tool, so each declaration was read the same day from the source file named after it at
// https://raw.githubusercontent.com/ml-explore/mlx-swift/main/ ):
// - loadArrays(url:stream:)
//   https://swiftpackageindex.com/ml-explore/mlx-swift/documentation/mlx/loadarrays(url:stream:)
//   Source/MLX/IO.swift: "public func loadArrays(url: URL, stream: StreamOrDevice = .cpu) throws
//   -> [String: MLXArray]", "Load dictionary of MLXArray from a safetensors file."
// - ModuleParameters
//   https://swiftpackageindex.com/ml-explore/mlx-swift/documentation/mlxnn/moduleparameters
//   Source/MLXNN/Module.swift: "public typealias ModuleParameters = NestedDictionary<String, MLXArray>"
// - NestedDictionary.unflattened(_:)
//   https://swiftpackageindex.com/ml-explore/mlx-swift/documentation/mlx/nesteddictionary/unflattened(_:)
//   Source/MLX/Nested.swift: "static public func unflattened(_ flat: [String: Element]) -> Self
//   where Key == String"; keys are split on "." and a segment that parses as an Int makes an
//   array level sized to the largest index plus one ("if Int(first) != nil").
// - Module.update(parameters:verify:path:modulePath:) and VerifyUpdate
//   https://swiftpackageindex.com/ml-explore/mlx-swift/documentation/mlxnn/module/update(parameters:verify:path:modulepath:)
//   https://swiftpackageindex.com/ml-explore/mlx-swift/documentation/mlxnn/verifyupdate
//   Source/MLXNN/Module.swift: "open func update(parameters: ModuleParameters, verify: VerifyUpdate,
//   path: [String] = [], modulePath: [String] = []) throws -> Self"; VerifyUpdate options
//   noUnusedKeys, allModelKeysSet, shapeMismatch, all, none; errors are the cases of
//   "public enum UpdateError: Error" (keyNotFound, mismatchedSize, unhandledKeys and the rest).
// - MLXArray.shape, MLXArray.dtype, MLXArray.size
//   https://swiftpackageindex.com/ml-explore/mlx-swift/documentation/mlx/mlxarray
//   Source/MLX/MLXArray.swift: "final public var shape: [Int]", "final public var dtype: DType",
//   "final public var size: Int" ("Total number of elements in the array").
// - DType
//   https://swiftpackageindex.com/ml-explore/mlx-swift/documentation/mlx/dtype
//   Source/MLX/DType.swift: cases bool, uint8, uint16, uint32, uint64, int8, int16, int32,
//   int64, float16, float32, bfloat16, complex64, float64. Fetched again November 4, 2026
//   (https://raw.githubusercontent.com/ml-explore/mlx-swift/main/Source/MLX/DType.swift):
//   "public enum DType: Hashable, Sendable, CaseIterable", so TensorMismatch below may be
//   Equatable and Sendable while holding a DType.
// - eval(_:)
//   https://swiftpackageindex.com/ml-explore/mlx-swift/documentation/mlx/eval(_:)
//   Source/MLX/Transforms+Eval.swift: "public func eval(_ arrays: some Collection<MLXArray>)".
//   Fetched again November 4, 2026
//   (https://raw.githubusercontent.com/ml-explore/mlx-swift/main/Source/MLX/Transforms+Eval.swift):
//   the other overloads are "eval(_ arrays: MLXArray...)", "eval(_ values: Any...)" and
//   "eval(_ values: some Sequence<Any>)"; an [MLXArray] argument matches the non-variadic
//   Collection<MLXArray> form, which load(url:) calls.
// - MLXArray.init(_:_:dtype:) over Data, used by SafetensorsFileReader
//   Source/MLX/MLXArray+Init.swift (read November 4, 2026): "public convenience init(_ data:
//   Data, _ shape: (some Collection<Int>)? = [Int]?.none, dtype: DType)".
// - mlx/io/safetensors.cpp of MLX (read November 4, 2026): dtype_from_safetensor_str has no F64
//   branch and ends in "[safetensor] unsupported dtype", the reason SafetensorsFileReader exists.
// Foundation symbols: Data(contentsOf:options:) with .mappedIfSafe,
// JSONSerialization.jsonObject(with:options:), Bundle.url(forResource:withExtension:subdirectory:)
// (https://developer.apple.com/documentation/foundation/bundle/url(forresource:withextension:subdirectory:))
// and JSONDecoder.decode(_:from:)
// (https://developer.apple.com/documentation/foundation/jsondecoder/decode(_:from:)).
//
// SOLID. S: WeightManifest decodes and indexes the manifest; SafetensorsFileReader parses the
// file layout; SafetensorsLoader checks; LoadedWeights hands checked arrays to module trees;
// WeightLoadReport lists the outcome. D: SafetensorsLoader reads the file through the
// SafetensorsReading protocol, injected at init (SafetensorsFileReader by default,
// MLXSafetensorsReader over loadArrays as the alternative; the tests inject a reader that builds
// arrays from the manifest, so every check runs without the 73 MB file).
//
// WEIGHTSLOADING CONFORMANCE. Work item WI-20 declares the protocol in
// Sources/UEMACore/Protocols.swift ("public protocol WeightsLoading<Weights>", one associated type
// Weights and one requirement "func load(from url: URL) throws -> (weights: Weights, report:
// WeightsReport)"; WeightsReport holds fileURL, loadedKeys, skippedKeys and loadedElements). The
// extension at the end of this file sets Weights = LoadedWeights and forwards load(from:) to
// SafetensorsLoader.load(url:); LoadedWeights.weightsReport fills WeightsReport from the manifest:
// loadedKeys are the 334 keys of the ten acting-path components in manifest order, skippedKeys the
// other 468, loadedElements the elements of the loaded keys' arrays (9,926,358 parameter elements
// plus 65,537 buffer elements = 9,991,895 for the released file: 4 causal masks of [128, 128] and
// one scalar input_scale, manifest records). A missing or extra key throws before any report
// exists, as the protocol's doc comment requires.
//
// FIRST BUILD ON A MAC (this work item's acceptance; plan section 8 and README.md steps 1 to 4).
//  1. Steps 1 to 3 of apple/README.md "The Mac first-build sequence" have passed (weights fetched,
//     MuJoCo XCFramework built, `swift package resolve` exit 0).
//  2. `xcodebuild -scheme UEMAKit-Package test -destination 'platform=macOS'
//     -only-testing:UEMAModelTests/SafetensorsLoaderTests` : exit code 0.
//     Without the released file (every test but the last; 15 tests): the loader's expected-key set equals the
//     manifest's 802 keys; a removed key, an added key, a changed shape and a changed dtype are
//     each refused with the key named; the unflattened parameters of one Linear layer
//     (wellness.grounded.tactile_comfort.net.0: weight [32, 8], bias [32]) pass
//     Module.update(parameters:verify: .all); through `any WeightsLoading<LoadedWeights>` the
//     report reads 334 loaded keys, 468 skipped keys and 9,991,895 loaded elements.
//     With the file (testRealFileHasTheReceiptParameterCount, which skips when the file is
//     absent): the loaded parameter tensors hold 18,131,745 elements, the report reads 0 missing
//     and 0 shape mismatches, and all 802 tensors are present. The file is found at the
//     manifest's source.path under the repository root (five path components above this test
//     file) or at the path in the environment variable UEMA_WEIGHTS_PATH.
//     The synthetic-file tests (testFileReaderReadsF32F64AndBool and
//     testFileReaderRefusesBadFiles) write small safetensors files into the temporary folder and
//     check SafetensorsFileReader's arrays and refusals; the real-file test also checks that the
//     parsed header is 91,104 bytes with 802 entries (manifest source.header_bytes).
//  3. If `MLX.loadArrays` does not resolve, the MLX product is missing from the UEMAModel target
//     in Package.swift (it is declared there as mlx("MLX")); if `ModuleParameters` does not
//     resolve, MLXNN is missing. If `MLXArray.zeros(_:dtype:stream:)` in the tests or
//     `MLXArray(_:_:dtype:)` in SafetensorsFileReader does not resolve, read
//     Source/MLX/Factory.swift or Source/MLX/MLXArray+Init.swift of the resolved mlx-swift tag
//     for its current form. If the real-file test stops at an F64 array (float64 arrays exist on
//     the CPU only in MLX), the 49 F64 tensors are all in skipped components: record the message
//     in apple/FIRST_BUILD_RECORD.md before changing the reader.
//  4. Write the command, exit code and clock time (AM/PM Pacific) into apple/FIRST_BUILD_RECORD.md
//     (WI-41), and the element count the real-file test printed.

import Foundation
import MLX
import MLXNN
import UEMACore

// MARK: - Element types

/// The element types a safetensors header may name (the safetensors format's dtype strings),
/// the same table as BYTES_PER_ELEMENT in apple/scripts/dump_weight_manifest.py. The released
/// file uses four of them (manifest counts.dtypes): F32 on 703 tensors, BOOL on 49, F64 on 49,
/// I64 on 1.
public enum SafetensorsDType: String, Codable, CaseIterable, Sendable {
    case f64 = "F64"
    case f32 = "F32"
    case f16 = "F16"
    case bf16 = "BF16"
    case i64 = "I64"
    case i32 = "I32"
    case i16 = "I16"
    case i8 = "I8"
    case u8 = "U8"
    case bool = "BOOL"

    /// Bytes one element occupies in the file (safetensors format).
    public var bytesPerElement: Int {
        switch self {
        case .f64, .i64: return 8
        case .f32, .i32: return 4
        case .f16, .bf16, .i16: return 2
        case .i8, .u8, .bool: return 1
        }
    }

    /// The MLX element type loadArrays gives an array of this safetensors type (DType cases read
    /// from Source/MLX/DType.swift, see the header).
    public var mlxDType: DType {
        switch self {
        case .f64: return .float64
        case .f32: return .float32
        case .f16: return .float16
        case .bf16: return .bfloat16
        case .i64: return .int64
        case .i32: return .int32
        case .i16: return .int16
        case .i8: return .int8
        case .u8: return .uint8
        case .bool: return .bool
        }
    }
}

/// Whether a saved tensor is a learned parameter (676 tensors, 18,131,745 elements) or a buffer
/// (126 tensors, 168,674 elements: causal masks, scales, repair bookkeeping); the manifest's
/// `kind` field, taken from the receipt's checkpoint_inventory by dump_weight_manifest.py.
public enum TensorKind: String, Codable, Sendable {
    case parameter
    case buffer
}

// MARK: - The manifest

/// One record of the manifest's `tensors` list: one saved tensor of model.safetensors.
public struct ManifestTensor: Decodable, Equatable, Sendable {
    /// The safetensors key, "<component>.<state_dict key>" (checkpoint.py:383).
    public let key: String
    /// The dimensions as saved; [] is a scalar of 1 element.
    public let shape: [Int]
    /// The element type as the safetensors header names it.
    public let dtype: SafetensorsDType
    /// The product of the shape (1 for a scalar).
    public let elements: Int
    /// elements x bytes per element.
    public let bytes: Int
    /// The receipt component ("enc_vision", "forward_model", ...; component_of() in
    /// dump_weight_manifest.py).
    public let component: String
    /// Parameter or buffer.
    public let kind: TensorKind
    /// True when the component is one the Swift model loads (REQ-B-11's inference set).
    public let inferenceSet: Bool

    enum CodingKeys: String, CodingKey {
        case key, shape, dtype, elements, bytes, component, kind
        case inferenceSet = "inference_set"
    }

    /// The element count the shape implies (the product of its entries; 1 for []).
    public var shapeElements: Int { shape.reduce(1, *) }
}

/// The manifest's `source` block: which file the manifest describes.
public struct ManifestSource: Decodable, Equatable, Sendable {
    /// The file's path relative to the repository root.
    public let path: String
    /// The file's size in bytes (73,119,988 for the released file).
    public let bytes: Int
    /// The file's SHA-256 hex digest.
    public let sha256: String
    /// 8 plus the safetensors header length, in bytes.
    public let headerBytes: Int
    /// The model step of the checkpoint (494,025).
    public let modelStep: Int
    /// When the receipt beside the file was written (ISO 8601).
    public let receiptGeneratedAt: String
    /// The receipt's total_params (18,131,745).
    public let receiptTotalParams: Int

    enum CodingKeys: String, CodingKey {
        case path, bytes, sha256
        case headerBytes = "header_bytes"
        case modelStep = "model_step"
        case receiptGeneratedAt = "receipt_generated_at"
        case receiptTotalParams = "receipt_total_params"
    }
}

/// The manifest's `counts` block: totals over the tensor records.
public struct ManifestCounts: Decodable, Equatable, Sendable {
    public let tensors: Int
    public let parameterTensors: Int
    public let bufferTensors: Int
    public let parameterElements: Int
    public let bufferElements: Int
    public let inferenceSetParameterElements: Int
    /// Tensor count per safetensors dtype name.
    public let dtypes: [String: Int]

    enum CodingKeys: String, CodingKey {
        case tensors, dtypes
        case parameterTensors = "parameter_tensors"
        case bufferTensors = "buffer_tensors"
        case parameterElements = "parameter_elements"
        case bufferElements = "buffer_elements"
        case inferenceSetParameterElements = "inference_set_parameter_elements"
    }
}

/// One entry of the manifest's `components` block: the totals of one receipt component.
public struct ManifestComponent: Decodable, Equatable, Sendable {
    public let parameterElements: Int
    public let bufferElements: Int
    public let tensors: Int
    /// The receipt's component_params value for this component.
    public let receiptComponentParams: Int
    public let inferenceSet: Bool

    enum CodingKeys: String, CodingKey {
        case tensors
        case parameterElements = "parameter_elements"
        case bufferElements = "buffer_elements"
        case receiptComponentParams = "receipt_component_params"
        case inferenceSet = "inference_set"
    }
}

/// The decoded weight_manifest.json: every tensor of the released model.safetensors with its
/// key, shape, dtype, kind and component, plus the file's identity and the totals. Decoding
/// checks the schema name, refuses duplicate keys, and checks each record's `elements` against
/// its shape and its `bytes` against elements x bytes per element.
public struct WeightManifest: Decodable, Equatable, Sendable {
    /// The schema name dump_weight_manifest.py writes (its SCHEMA constant).
    public static let expectedSchema = "uema_weight_manifest_v1"
    /// The bundled resource's name and extension (Sources/UEMAModel/Resources/weight_manifest.json).
    public static let resourceName = "weight_manifest"
    public static let resourceExtension = "json"
    public static let resourceSubdirectory = "Resources"

    public let schema: String
    public let generatedAt: String
    public let source: ManifestSource
    public let counts: ManifestCounts
    public let components: [String: ManifestComponent]
    public let tensors: [ManifestTensor]
    /// The records indexed by key, built at decode time.
    public let byKey: [String: ManifestTensor]

    enum CodingKeys: String, CodingKey {
        case schema, source, counts, components, tensors
        case generatedAt = "generated_at"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let schema = try container.decode(String.self, forKey: .schema)
        guard schema == WeightManifest.expectedSchema else {
            throw SafetensorsLoaderError.manifestSchema(found: schema, expected: WeightManifest.expectedSchema)
        }
        self.schema = schema
        self.generatedAt = try container.decode(String.self, forKey: .generatedAt)
        self.source = try container.decode(ManifestSource.self, forKey: .source)
        let counts = try container.decode(ManifestCounts.self, forKey: .counts)
        self.counts = counts
        self.components = try container.decode([String: ManifestComponent].self, forKey: .components)
        let tensors = try container.decode([ManifestTensor].self, forKey: .tensors)
        self.tensors = tensors
        var index: [String: ManifestTensor] = [:]
        var problems: [String] = []
        for tensor in tensors {
            if index[tensor.key] != nil {
                problems.append("key \(tensor.key) appears twice")
            }
            index[tensor.key] = tensor
            if tensor.elements != tensor.shapeElements {
                problems.append("key \(tensor.key): elements \(tensor.elements) but shape \(tensor.shape) holds \(tensor.shapeElements)")
            }
            let expectedBytes = tensor.elements * tensor.dtype.bytesPerElement
            if tensor.bytes != expectedBytes {
                problems.append("key \(tensor.key): bytes \(tensor.bytes) but \(tensor.elements) x \(tensor.dtype.bytesPerElement) bytes = \(expectedBytes)")
            }
        }
        if counts.tensors != tensors.count {
            problems.append("counts.tensors \(counts.tensors) but \(tensors.count) records")
        }
        let parameterSum = tensors.filter { $0.kind == .parameter }.reduce(0) { $0 + $1.elements }
        if counts.parameterElements != parameterSum {
            problems.append("counts.parameter_elements \(counts.parameterElements) but the parameter records sum to \(parameterSum)")
        }
        if !problems.isEmpty {
            throw SafetensorsLoaderError.manifestInconsistent(problems)
        }
        self.byKey = index
    }

    /// Decode a manifest from JSON bytes (JSONDecoder.decode(_:from:)).
    public init(data: Data) throws {
        self = try JSONDecoder().decode(WeightManifest.self, from: data)
    }

    /// The manifest copied into the module's resource bundle by the Package.swift `copy` rule
    /// (Bundle.url(forResource:withExtension:subdirectory:)); `bundle` defaults to
    /// UEMAModelModule.resourceBundle and the tests pass it through.
    public static func bundled(in bundle: Bundle = UEMAModelModule.resourceBundle) throws -> WeightManifest {
        guard let url = bundle.url(forResource: resourceName, withExtension: resourceExtension,
                                   subdirectory: resourceSubdirectory) else {
            throw SafetensorsLoaderError.manifestNotFound(
                resource: "\(resourceSubdirectory)/\(resourceName).\(resourceExtension)",
                bundle: bundle.bundlePath)
        }
        return try WeightManifest(data: try Data(contentsOf: url))
    }

    /// Every tensor key (802 in the released manifest).
    public var keys: Set<String> { Set(byKey.keys) }

    /// The sum of the parameter records' elements (18,131,745 in the released manifest).
    public var parameterElements: Int {
        tensors.filter { $0.kind == .parameter }.reduce(0) { $0 + $1.elements }
    }

    /// The records of one receipt component, in key order.
    public func tensors(inComponent component: String) -> [ManifestTensor] {
        tensors.filter { $0.component == component }
    }

    /// The receipt component a safetensors key belongs to: the key's first segment, except that
    /// "encoder_registry.encoders.<sense>." keys belong to "enc_<sense>" and any other
    /// "encoder_registry." key to "enc_shared" (component_of() in dump_weight_manifest.py;
    /// lem_model.py:1398-1402).
    public static func component(ofKey key: String) -> String {
        let parts = key.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        guard let first = parts.first else { return key }
        if first == "encoder_registry" {
            if parts.count >= 4, parts[1] == "encoders" {
                return "enc_" + parts[2]
            }
            return "enc_shared"
        }
        return first
    }
}

// MARK: - Components of the acting path

/// The ten receipt components the Swift model loads (REQ-B-11: the six encoders, the forward
/// model, the prediction heads, the motor controller and wellness; 9,926,358 parameters in all,
/// manifest counts.inference_set_parameter_elements). The raw value is the manifest's component
/// name; the other 14 components of the manifest are listed and skipped by name.
public enum WeightComponent: String, CaseIterable, Codable, Sendable {
    case visionEncoder = "enc_vision"
    case tactileEncoder = "enc_tactile"
    case audioEncoder = "enc_audio"
    case proprioceptionEncoder = "enc_proprioception"
    case motorEncoder = "enc_motor"
    case vestibularEncoder = "enc_vestibular"
    case forwardModel = "forward_model"
    case predictionHeads = "prediction_heads"
    case motorController = "motor_controller"
    case wellness = "wellness"

    /// The sense an encoder component serves (the model registry's name: vision, tactile,
    /// audio, proprioception, motor, vestibular); nil for the four non-encoder components.
    public var senseName: String? {
        switch self {
        case .visionEncoder: return "vision"
        case .tactileEncoder: return "tactile"
        case .audioEncoder: return "audio"
        case .proprioceptionEncoder: return "proprioception"
        case .motorEncoder: return "motor"
        case .vestibularEncoder: return "vestibular"
        case .forwardModel, .predictionHeads, .motorController, .wellness: return nil
        }
    }

    /// The text every key of this component begins with, which parameters(for:) removes so the
    /// remainder is the PyTorch state_dict key of the component's own module
    /// ("encoder_registry.encoders.vision." for the vision encoder, checkpoint.py:383 and
    /// lem_model.py:1398; "<component>." for the rest).
    public var keyPrefix: String {
        if let sense = senseName {
            return "encoder_registry.encoders." + sense + "."
        }
        return rawValue + "."
    }
}

// MARK: - Errors

/// One tensor whose shape or element type in the file differs from the manifest.
public struct TensorMismatch: Equatable, Sendable, CustomStringConvertible {
    public let key: String
    public let manifestShape: [Int]
    public let fileShape: [Int]
    public let manifestDType: SafetensorsDType
    public let fileDType: DType

    public init(key: String, manifestShape: [Int], fileShape: [Int],
                manifestDType: SafetensorsDType, fileDType: DType) {
        self.key = key
        self.manifestShape = manifestShape
        self.fileShape = fileShape
        self.manifestDType = manifestDType
        self.fileDType = fileDType
    }

    public var shapeDiffers: Bool { manifestShape != fileShape }
    public var dtypeDiffers: Bool { manifestDType.mlxDType != fileDType }

    public var description: String {
        var parts: [String] = []
        if shapeDiffers {
            parts.append("shape \(fileShape) in the file, \(manifestShape) in the manifest")
        }
        if dtypeDiffers {
            parts.append("dtype \(fileDType) in the file, \(manifestDType.rawValue) in the manifest")
        }
        return "\(key): " + parts.joined(separator: "; ")
    }
}

/// Every way the loader refuses; each case names the keys or values involved.
public enum SafetensorsLoaderError: Error, CustomStringConvertible {
    /// The manifest resource is not in the bundle.
    case manifestNotFound(resource: String, bundle: String)
    /// The manifest's schema field is not the one this loader reads.
    case manifestSchema(found: String, expected: String)
    /// The manifest disagrees with itself (duplicate key, elements against shape, bytes
    /// against elements, counts against records); each sentence names the key.
    case manifestInconsistent([String])
    /// The file's key set differs from the manifest's: `missing` are manifest keys absent from
    /// the file, `extra` are file keys absent from the manifest; both sorted.
    case keysDiffer(missing: [String], extra: [String])
    /// Keys present on both sides whose shape or dtype differs, in key order.
    case tensorsDiffer([TensorMismatch])
    /// Module.update(parameters:verify:) refused one component's parameters; `underlying` is
    /// MLXNN's UpdateError.
    case moduleUpdate(component: WeightComponent, underlying: Error)

    public var description: String {
        switch self {
        case .manifestNotFound(let resource, let bundle):
            return "weight manifest \(resource) is not in the bundle at \(bundle)"
        case .manifestSchema(let found, let expected):
            return "weight manifest schema is \(found); this loader reads \(expected)"
        case .manifestInconsistent(let problems):
            return "weight manifest disagrees with itself:\n" + problems.joined(separator: "\n")
        case .keysDiffer(let missing, let extra):
            var lines: [String] = []
            lines.append("\(missing.count) manifest keys are missing from the file and \(extra.count) file keys are not in the manifest")
            lines.append(contentsOf: missing.map { "missing: \($0)" })
            lines.append(contentsOf: extra.map { "extra: \($0)" })
            return lines.joined(separator: "\n")
        case .tensorsDiffer(let mismatches):
            return "\(mismatches.count) tensors differ from the manifest:\n"
                + mismatches.map { $0.description }.joined(separator: "\n")
        case .moduleUpdate(let component, let underlying):
            return "parameters of \(component.rawValue) were refused by the module tree: \(underlying)"
        }
    }
}

// MARK: - Reading the file

/// Reads a safetensors file into a dictionary from tensor name to array. SafetensorsLoader
/// depends on this protocol, not on MLX's file function, so a test can hand it arrays built
/// from the manifest.
public protocol SafetensorsReading {
    /// Every tensor of the file at `url`, keyed by its safetensors name.
    func readArrays(url: URL) throws -> [String: MLXArray]
}

/// A reader over MLX.loadArrays(url:) (Source/MLX/IO.swift; default stream .cpu). It refuses
/// the released model.safetensors: MLX's safetensors parser (mlx/io/safetensors.cpp,
/// dtype_from_safetensor_str, read November 4, 2026 at
/// https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/io/safetensors.cpp) accepts F32,
/// F16, BF16, I64, I32, I16, I8, U64, U32, U16, U8, BOOL, C64, F8_E4M3 and F8_E8M0 and throws
/// "[safetensor] unsupported dtype" for any other string, and the released file holds 49 F64
/// tensors (manifest counts.dtypes; all of them buffers of three training-only components:
/// liveness_repair 42, persistent_memory 6, heads_all_slot_path 1). It stays for files without
/// F64 tensors; the loader's default reader is SafetensorsFileReader.
public struct MLXSafetensorsReader: SafetensorsReading {
    public init() {}

    public func readArrays(url: URL) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: url)
    }
}

/// Every way SafetensorsFileReader refuses a file; each case names the byte counts or the key.
public enum SafetensorsFileError: Error, CustomStringConvertible {
    /// The file is shorter than its own 8-byte header-length field says.
    case truncatedHeader(fileBytes: Int, declaredHeaderBytes: UInt64)
    /// The header bytes are not a JSON object.
    case headerNotJSON
    /// One header entry lacks dtype, shape or data_offsets, or holds a negative number.
    case malformedEntry(key: String)
    /// One header entry names an element type outside SafetensorsDType.
    case unknownDType(key: String, dtype: String)
    /// One entry's byte range disagrees with its shape x bytes per element, or runs past the
    /// end of the data section.
    case badByteRange(key: String, start: Int, end: Int, expectedBytes: Int, dataBytes: Int)

    public var description: String {
        switch self {
        case .truncatedHeader(let fileBytes, let declared):
            return "safetensors header declares \(declared) bytes but the file holds \(fileBytes) bytes"
        case .headerNotJSON:
            return "safetensors header is not a JSON object"
        case .malformedEntry(let key):
            return "safetensors header entry \(key) lacks dtype, shape or data_offsets, or holds a negative number"
        case .unknownDType(let key, let dtype):
            return "safetensors header entry \(key) names dtype \(dtype), which this reader does not read"
        case .badByteRange(let key, let start, let end, let expectedBytes, let dataBytes):
            return "safetensors header entry \(key): data_offsets [\(start), \(end)] against \(expectedBytes) bytes of shape x dtype, data section \(dataBytes) bytes"
        }
    }
}

/// One tensor entry of a safetensors header: key, element type, shape and its byte range
/// within the data section (the bytes after the header).
public struct SafetensorsHeaderEntry: Equatable, Sendable {
    public let key: String
    public let dtype: SafetensorsDType
    public let shape: [Int]
    /// Offsets counted from the first byte after the header, end exclusive.
    public let dataRange: Range<Int>
}

/// A parsed safetensors header: its size in bytes (8 for the length field plus the JSON text;
/// 91,104 bytes for the released file, manifest source.header_bytes) and its tensor entries
/// sorted by key ("__metadata__" is not a tensor and is left out).
public struct SafetensorsHeader: Equatable, Sendable {
    public let headerBytes: Int
    public let entries: [SafetensorsHeaderEntry]
}

/// The production reader. It parses the safetensors layout with Foundation and builds each
/// MLXArray from the tensor's bytes, so F64 tensors load (MLX.loadArrays refuses them, see
/// MLXSafetensorsReader). Layout (the safetensors format; the same reading as
/// safetensors_header_bytes() in apple/scripts/dump_weight_manifest.py:118-126, which unpacks
/// "<Q" from the first 8 bytes): bytes 0..<8 hold the header's length N as a little-endian
/// unsigned 64-bit integer; bytes 8..<8+N hold a JSON object mapping each key to {"dtype",
/// "shape", "data_offsets": [start, end]} plus an optional "__metadata__" entry; the offsets
/// count from byte 8+N. Each array is MLXArray(_:_:dtype:) over the tensor's bytes
/// (Source/MLX/MLXArray+Init.swift of mlx-swift, read November 4, 2026 at
/// https://raw.githubusercontent.com/ml-explore/mlx-swift/main/Source/MLX/MLXArray+Init.swift:
/// "public convenience init(_ data: Data, _ shape: (some Collection<Int>)? = [Int]?.none,
/// dtype: DType)"). Element bytes are taken as little-endian, the byte order of the safetensors
/// format and of Apple silicon.
public struct SafetensorsFileReader: SafetensorsReading {
    public init() {}

    public func readArrays(url: URL) throws -> [String: MLXArray] {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let header = try Self.parseHeader(data)
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(header.entries.count)
        let dataStart = data.startIndex + header.headerBytes
        for entry in header.entries {
            let lower = dataStart + entry.dataRange.lowerBound
            let upper = dataStart + entry.dataRange.upperBound
            let bytes = data.subdata(in: lower..<upper)
            out[entry.key] = MLXArray(bytes, entry.shape, dtype: entry.dtype.mlxDType)
        }
        return out
    }

    /// The header of the file at `url`.
    public func readHeader(url: URL) throws -> SafetensorsHeader {
        try Self.parseHeader(try Data(contentsOf: url, options: .mappedIfSafe))
    }

    /// Parse the header of a whole safetensors file held in `data`, checking every entry's dtype
    /// and that its byte range equals shape elements x bytes per element and lies inside the
    /// data section.
    public static func parseHeader(_ data: Data) throws -> SafetensorsHeader {
        let fileBytes = data.count
        guard fileBytes >= 8 else {
            throw SafetensorsFileError.truncatedHeader(fileBytes: fileBytes, declaredHeaderBytes: 8)
        }
        var declared: UInt64 = 0
        for (index, byte) in data.prefix(8).enumerated() {
            declared |= UInt64(byte) << UInt64(8 * index)
        }
        guard declared <= UInt64(fileBytes - 8) else {
            throw SafetensorsFileError.truncatedHeader(fileBytes: fileBytes, declaredHeaderBytes: declared &+ 8)
        }
        let headerBytes = 8 + Int(declared)
        let jsonData = data.subdata(in: (data.startIndex + 8)..<(data.startIndex + headerBytes))
        guard let object = try? JSONSerialization.jsonObject(with: jsonData, options: []),
              let table = object as? [String: Any] else {
            throw SafetensorsFileError.headerNotJSON
        }
        let dataBytes = fileBytes - headerBytes
        var entries: [SafetensorsHeaderEntry] = []
        for key in table.keys.sorted() where key != "__metadata__" {
            guard let fields = table[key] as? [String: Any],
                  let dtypeName = fields["dtype"] as? String,
                  let shape = intArray(fields["shape"]),
                  let offsets = intArray(fields["data_offsets"]),
                  offsets.count == 2,
                  !shape.contains(where: { $0 < 0 }),
                  !offsets.contains(where: { $0 < 0 }) else {
                throw SafetensorsFileError.malformedEntry(key: key)
            }
            guard let dtype = SafetensorsDType(rawValue: dtypeName) else {
                throw SafetensorsFileError.unknownDType(key: key, dtype: dtypeName)
            }
            let start = offsets[0]
            let end = offsets[1]
            let expectedBytes = shape.reduce(1, *) * dtype.bytesPerElement
            guard start <= end, end - start == expectedBytes, end <= dataBytes else {
                throw SafetensorsFileError.badByteRange(key: key, start: start, end: end,
                                                        expectedBytes: expectedBytes, dataBytes: dataBytes)
            }
            entries.append(SafetensorsHeaderEntry(key: key, dtype: dtype, shape: shape, dataRange: start..<end))
        }
        return SafetensorsHeader(headerBytes: headerBytes, entries: entries)
    }

    /// A JSON number list as [Int] (JSONSerialization gives NSNumber elements).
    private static func intArray(_ value: Any?) -> [Int]? {
        if let ints = value as? [Int] {
            return ints
        }
        if let numbers = value as? [NSNumber] {
            return numbers.map { $0.intValue }
        }
        return nil
    }
}

// MARK: - The loaded weights

/// One line of the load report: one manifest tensor and whether the acting path loads it.
public struct WeightLoadLine: Equatable, Sendable, CustomStringConvertible {
    public let key: String
    public let component: String
    public let kind: TensorKind
    public let shape: [Int]
    public let dtype: SafetensorsDType
    public let elements: Int
    /// True when the component is in the acting path (WeightComponent), false when skipped.
    public let loaded: Bool

    public var description: String {
        "\(loaded ? "loaded " : "skipped") \(key) \(shape) \(dtype.rawValue) \(kind.rawValue) \(elements) elements"
    }
}

/// The outcome of one load, one line per manifest tensor (REQ-B-11: "Every manifest tensor is
/// reported loaded or skipped by name: 0 missing, 0 shape mismatches").
public struct WeightLoadReport: Equatable, Sendable, CustomStringConvertible {
    public let lines: [WeightLoadLine]
    /// Manifest keys absent from the file; 0 after a successful load.
    public let missing: Int
    /// Keys whose file shape differed from the manifest; 0 after a successful load.
    public let shapeMismatches: Int

    public var loadedTensors: Int { lines.filter { $0.loaded }.count }
    public var skippedTensors: Int { lines.filter { !$0.loaded }.count }
    public var loadedParameterElements: Int {
        lines.filter { $0.loaded && $0.kind == .parameter }.reduce(0) { $0 + $1.elements }
    }
    public var skippedParameterElements: Int {
        lines.filter { !$0.loaded && $0.kind == .parameter }.reduce(0) { $0 + $1.elements }
    }

    public var description: String {
        var out = lines.map { $0.description }
        out.append("loaded \(loadedTensors) tensors (\(loadedParameterElements) parameter elements), skipped \(skippedTensors) tensors (\(skippedParameterElements) parameter elements), \(missing) missing, \(shapeMismatches) shape mismatches")
        return out.joined(separator: "\n")
    }
}

/// The checked arrays of one model.safetensors with the manifest they were checked against.
public struct LoadedWeights {
    /// Where the file was read from.
    public let url: URL
    public let manifest: WeightManifest
    /// Every tensor of the file, keyed by its safetensors name; the key set equals the
    /// manifest's and every shape and dtype equals its record.
    public let arrays: [String: MLXArray]

    public init(url: URL, manifest: WeightManifest, arrays: [String: MLXArray]) {
        self.url = url
        self.manifest = manifest
        self.arrays = arrays
    }

    /// The element count of the loaded parameter tensors, measured from the arrays
    /// (MLXArray.size summed over the manifest's parameter keys): 18,131,745 for the released
    /// file, the receipt's total_params.
    public var parameterElements: Int {
        manifest.tensors.filter { $0.kind == .parameter }.reduce(0) { $0 + (arrays[$1.key]?.size ?? 0) }
    }

    /// The element count of every loaded array (parameters and buffers): 18,300,419 for the
    /// released file (manifest counts.parameter_elements + counts.buffer_elements).
    public var totalElements: Int {
        arrays.values.reduce(0) { $0 + $1.size }
    }

    /// The array of one safetensors key, or nil when the manifest has no such key.
    public func array(forKey key: String) -> MLXArray? {
        arrays[key]
    }

    /// The arrays whose key begins with `keyPrefix`, re-keyed by the remainder of the key.
    /// `includeBuffers` false (the default) keeps the parameter records only, so a module tree
    /// that declares no buffer property passes `verify: .noUnusedKeys`; true adds the buffers
    /// (forward_model.blocks.<n>.causal_mask, prediction_heads.heads.vision.mlp.0.input_scale
    /// in the acting path).
    public func flatParameters(keyPrefix: String, includeBuffers: Bool = false) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for tensor in manifest.tensors where tensor.key.hasPrefix(keyPrefix) {
            if tensor.kind == .buffer && !includeBuffers { continue }
            guard let array = arrays[tensor.key] else { continue }
            out[String(tensor.key.dropFirst(keyPrefix.count))] = array
        }
        return out
    }

    /// flatParameters(keyPrefix:includeBuffers:) nested through ModuleParameters.unflattened:
    /// "blocks.0.ffn.0.weight" becomes blocks -> index 0 -> ffn -> index 0 -> weight, the shape
    /// Module.update(parameters:verify:) walks (Source/MLX/Nested.swift, see the header).
    public func parameters(keyPrefix: String, includeBuffers: Bool = false) -> ModuleParameters {
        ModuleParameters.unflattened(flatParameters(keyPrefix: keyPrefix, includeBuffers: includeBuffers))
    }

    /// One acting-path component's arrays, keyed by the PyTorch state_dict key of the
    /// component's module (the component's keyPrefix removed), nested for Module.update.
    public func parameters(for component: WeightComponent, includeBuffers: Bool = false) -> ModuleParameters {
        parameters(keyPrefix: component.keyPrefix, includeBuffers: includeBuffers)
    }

    /// One acting-path component's buffer arrays alone, keyed by the state_dict key, for a
    /// module that rebuilds its buffers from them rather than declaring them as parameters.
    public func buffers(for component: WeightComponent) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for tensor in manifest.tensors(inComponent: component.rawValue) where tensor.kind == .buffer {
            guard let array = arrays[tensor.key] else { continue }
            out[String(tensor.key.dropFirst(component.keyPrefix.count))] = array
        }
        return out
    }

    /// Every manifest tensor as loaded (its component is a WeightComponent) or skipped, by name.
    public var report: WeightLoadReport {
        let lines = manifest.tensors.map { tensor in
            WeightLoadLine(key: tensor.key, component: tensor.component, kind: tensor.kind,
                           shape: tensor.shape, dtype: tensor.dtype, elements: tensor.elements,
                           loaded: WeightComponent(rawValue: tensor.component) != nil)
        }
        let missing = manifest.tensors.filter { arrays[$0.key] == nil }.count
        let shapeMismatches = manifest.tensors.filter { tensor in
            guard let array = arrays[tensor.key] else { return false }
            return array.shape != tensor.shape
        }.count
        return WeightLoadReport(lines: lines, missing: missing, shapeMismatches: shapeMismatches)
    }
}

// MARK: - The loader

/// Reads model.safetensors through a SafetensorsReading, checks every key, shape and dtype
/// against the manifest, and hands checked arrays to module trees.
public final class SafetensorsLoader {
    public let manifest: WeightManifest
    private let reader: SafetensorsReading

    /// `manifest` is the decoded weight_manifest.json; `reader` reads the file (the production
    /// reader, SafetensorsFileReader, parses the safetensors layout and reads the F64 tensors
    /// that MLX.loadArrays refuses).
    public init(manifest: WeightManifest, reader: SafetensorsReading = SafetensorsFileReader()) {
        self.manifest = manifest
        self.reader = reader
    }

    /// A loader over the bundled manifest and the production reader.
    public static func bundled(in bundle: Bundle = UEMAModelModule.resourceBundle,
                               reader: SafetensorsReading = SafetensorsFileReader()) throws -> SafetensorsLoader {
        SafetensorsLoader(manifest: try WeightManifest.bundled(in: bundle), reader: reader)
    }

    /// The keys the file must hold: the manifest's 802 keys, no more and no fewer.
    public var expectedKeys: Set<String> { manifest.keys }

    /// Read the file, check it against the manifest, evaluate the arrays and return them.
    /// Throws SafetensorsLoaderError.keysDiffer or .tensorsDiffer naming every offending key,
    /// or the reader's own error when the file cannot be read.
    public func load(url: URL) throws -> LoadedWeights {
        let arrays = try reader.readArrays(url: url)
        try check(arrays)
        eval(Array(arrays.values))
        return LoadedWeights(url: url, manifest: manifest, arrays: arrays)
    }

    /// Compare a dictionary of arrays with the manifest: first the key sets (missing and extra
    /// keys, both reported in one error), then, for every key on both sides, the shape and the
    /// dtype (every mismatch reported in one error). Returns normally when all three agree.
    public func check(_ arrays: [String: MLXArray]) throws {
        let fileKeys = Set(arrays.keys)
        let expected = expectedKeys
        let missing = expected.subtracting(fileKeys).sorted()
        let extra = fileKeys.subtracting(expected).sorted()
        if !missing.isEmpty || !extra.isEmpty {
            throw SafetensorsLoaderError.keysDiffer(missing: missing, extra: extra)
        }
        var mismatches: [TensorMismatch] = []
        for tensor in manifest.tensors {
            guard let array = arrays[tensor.key] else { continue }
            let candidate = TensorMismatch(key: tensor.key, manifestShape: tensor.shape, fileShape: array.shape,
                                           manifestDType: tensor.dtype, fileDType: array.dtype)
            if candidate.shapeDiffers || candidate.dtypeDiffers {
                mismatches.append(candidate)
            }
        }
        if !mismatches.isEmpty {
            throw SafetensorsLoaderError.tensorsDiffer(mismatches)
        }
    }

    /// Replace `module`'s parameters with one component's loaded arrays through
    /// Module.update(parameters:verify:), with `verify` .all by default (every loaded key used,
    /// every module key set, every shape equal). An UpdateError is rethrown as
    /// SafetensorsLoaderError.moduleUpdate naming the component.
    public func apply(_ weights: LoadedWeights, to module: Module, component: WeightComponent,
                      includeBuffers: Bool = false, verify: VerifyUpdate = .all) throws {
        do {
            _ = try module.update(parameters: weights.parameters(for: component, includeBuffers: includeBuffers),
                                  verify: verify)
        } catch {
            throw SafetensorsLoaderError.moduleUpdate(component: component, underlying: error)
        }
    }
}

// MARK: - The UEMACore contract

extension ManifestTensor {
    /// This record as UEMACore's WeightsManifestEntry (Protocols.swift, WI-20), whose dtype and
    /// kind are the manifest's strings ("F32", "parameter", ...).
    public var manifestEntry: WeightsManifestEntry {
        WeightsManifestEntry(key: key, shape: shape, dtype: dtype.rawValue, elements: elements,
                             bytes: bytes, component: component, kind: kind.rawValue,
                             inferenceSet: inferenceSet)
    }
}

extension LoadedWeights {
    /// The keys of the acting-path components (a WeightComponent raw value), in manifest order:
    /// 334 keys for the released file.
    public var loadedKeys: [String] {
        manifest.tensors.filter { WeightComponent(rawValue: $0.component) != nil }.map { $0.key }
    }

    /// The keys of the 14 training-only components, in manifest order: 468 keys for the
    /// released file.
    public var skippedKeys: [String] {
        manifest.tensors.filter { WeightComponent(rawValue: $0.component) == nil }.map { $0.key }
    }

    /// The elements of the loaded keys' arrays (MLXArray.size summed, parameters and buffers):
    /// 9,991,895 for the released file (9,926,358 parameter elements, manifest
    /// counts.inference_set_parameter_elements, plus 65,537 buffer elements).
    public var loadedElements: Int {
        loadedKeys.reduce(0) { $0 + (arrays[$1]?.size ?? 0) }
    }

    /// The outcome in UEMACore's WeightsReport form (plan row WI-31, REQ-B-11).
    public var weightsReport: WeightsReport {
        WeightsReport(fileURL: url, loadedKeys: loadedKeys, skippedKeys: skippedKeys,
                      loadedElements: loadedElements)
    }
}

/// SafetensorsLoader fulfils UEMACore's WeightsLoading (Protocols.swift, WI-20): the session
/// (WI-32) and the model (WI-35, WI-36) take `any WeightsLoading<LoadedWeights>` or a generic
/// parameter, never this class by name.
extension SafetensorsLoader: WeightsLoading {
    public typealias Weights = LoadedWeights

    /// Reads and checks the file (load(url:)) and returns the checked arrays with the report;
    /// a missing or extra key, a shape or a dtype difference throws SafetensorsLoaderError
    /// naming every offending key.
    public func load(from url: URL) throws -> (weights: LoadedWeights, report: WeightsReport) {
        let weights = try load(url: url)
        return (weights: weights, report: weights.weightsReport)
    }
}
