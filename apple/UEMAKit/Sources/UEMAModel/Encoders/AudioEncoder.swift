// AudioEncoder.swift: the audio sense's encoder of the released model in MLX Swift (work item WI-35
// of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-9). Nothing in
// this file was compiled on the machine that wrote it (no Swift toolchain on Windows).
//
// INPUT: 256 spectrum bins x 2 ears on a 0 to 1 scale [256, 2] and 2 spatial numbers [2]
//   (UEMATokenizer's AudioTokenizer, WI-30; the fixture's input/audio/spectral [1, 256, 2] and
//   input/audio/spatial [1, 2]).
// OPERATION (lem/model/encoders/conv1d_stack.py, Conv1DStackEncoder, read November 4, 2026):
//   1. Spectral branch: 4 Conv1d layers (conv_channels [192] x (2 + encoder_layers 2), size
//      overrides of mimo_bridge/train_mimo.py) of kernel 3, padding 1, no bias, each followed by
//      GELU (conv1d_stack.py:214-225; Sequential indices 0, 3, 6, 9 with GELU and Dropout between);
//      the mean over the 256 bins (AdaptiveAvgPool1d(1), conv1d_stack.py:228, 258); spectral_proj
//      Linear(192 -> 192, no bias). PyTorch's Conv1d reads [B, channels, bins] and stores its
//      weight [out, in, kernel]; MLX's conv1d reads [B, bins, channels] and a weight
//      [out, kernel, in], so the input is used as given and each weight is transposed (0, 2, 1).
//   2. Frequency-band path (freq_band_path on: band_proj.weight [192, 32] is in the manifest): the
//      raw spectrum's mean over 16 contiguous bands of 16 bins per ear, laid band by band with the
//      2 ears inside each band (32 numbers), through band_proj Linear(32 -> 192, no bias), added to
//      the spectral branch (conv1d_stack.py:260-268).
//   3. Spatial branch: spatial_proj Linear(2 -> 192, no bias) (conv1d_stack.py:232, 271).
//   4. Combine "gate" (lem/master_config.yaml:134): g = sigmoid(gate(cat[s, p])) with gate
//      Linear(384 -> 192, no bias); combined = g s + (1 - g) p (conv1d_stack.py:278-280).
//   5. norm LayerNorm(192); proj is the identity (conv1d_stack.py:242-244, 284-286).
// OUTPUT: [B, 1, 192] before the shared output LayerNorm (the fixture's encoder_output/audio).
// PARAMETERS: 1,152 + 3 x 110,592 + 36,864 + 6,144 + 384 + 73,728 + 384 = 450,432 (manifest
//   component enc_audio).
// FIRST BUILD ON A MAC: see TransformerCore.swift, steps 1 to 4. A conv1d argument-order error is
//   the first thing to check here: MLX conv1d (Ops.swift, mlx-swift main, read November 4, 2026)
//   is "conv1d(_ array: MLXArray, _ weight: MLXArray, stride: Int = 1, padding: Int = 0,
//   dilation: Int = 1, groups: Int = 1, stream:)", input [N, L, C_in], weight [C_out, K, C_in].

import Foundation
import MLX
import UEMACore

/// The audio encoder (Conv1DStackEncoder at the released sizes).
public struct AudioEncoder: SenseEncoding {
    /// Spectrum bins per ear: 256.
    public static let bins = 256
    /// Ears: 2.
    public static let ears = 2
    /// Spatial numbers: 2.
    public static let spatialNumbers = 2
    /// Width: 192.
    public static let width = 192
    /// Conv1d layers: 4, at Sequential indices 0, 3, 6, 9.
    public static let convIndices = [0, 3, 6, 9]
    /// Kernel size: 3.
    public static let kernel = 3
    /// Contiguous frequency bands of the band path: 16.
    public static let bands = 16
    /// Parameters: 450,432.
    public static let parameterCount = 450_432

    public let sense: SenseName = .audio
    /// The conv weights in MLX layout [out, kernel, in].
    let convWeights: [MLXArray]
    let spectralProjection: LinearLayer
    let bandProjection: LinearLayer
    let spatialProjection: LinearLayer
    let gate: LinearLayer
    let norm: LayerNormLayer

    /// The encoder stored under `source` (the released file seen through
    /// `scoped("encoder_registry.encoders.audio.")`).
    public init(source: any MLXTensorSource) throws {
        let width = AudioEncoder.width
        var weights: [MLXArray] = []
        var inputs = AudioEncoder.ears
        for index in AudioEncoder.convIndices {
            let torchWeight = try source.tensor("spectral_conv.\(index).weight",
                                                shape: [width, inputs, AudioEncoder.kernel])
            weights.append(torchWeight.transposed(0, 2, 1))
            inputs = width
        }
        convWeights = weights
        spectralProjection = try LinearLayer(source: source, prefix: "spectral_proj.", inputs: width,
                                             outputs: width, hasBias: false)
        bandProjection = try LinearLayer(source: source, prefix: "band_proj.",
                                         inputs: AudioEncoder.bands * AudioEncoder.ears, outputs: width,
                                         hasBias: false)
        spatialProjection = try LinearLayer(source: source, prefix: "spatial_proj.",
                                            inputs: AudioEncoder.spatialNumbers, outputs: width, hasBias: false)
        gate = try LinearLayer(source: source, prefix: "gate.", inputs: 2 * width, outputs: width, hasBias: false)
        norm = try LayerNormLayer(source: source, prefix: "norm.", width: width)
    }

    public var parameterCount: Int {
        convWeights.reduce(0) { $0 + $1.size } + spectralProjection.parameterCount + bandProjection.parameterCount
            + spatialProjection.parameterCount + gate.parameterCount + norm.parameterCount
    }

    /// The pooled vector [B, 1, 192] of a spectrum [B, 256, 2] and spatial numbers [B, 2].
    public func callAsFunction(spectrum: MLXArray, spatial: MLXArray) -> MLXArray {
        let batch = spectrum.shape[0]
        var convolved = spectrum
        for weight in convWeights {
            convolved = NeuralOps.gelu(conv1d(convolved, weight, stride: 1, padding: AudioEncoder.kernel / 2))
        }
        var spectral = spectralProjection(convolved.mean(axis: 1))
        let binsPerBand = AudioEncoder.bins / AudioEncoder.bands
        let bandMeans = spectrum
            .reshaped([batch, AudioEncoder.bands, binsPerBand, AudioEncoder.ears])
            .mean(axis: 2)
            .reshaped([batch, AudioEncoder.bands * AudioEncoder.ears])
        spectral = spectral + bandProjection(bandMeans)
        let spatialPart = spatialProjection(spatial)
        let g = sigmoid(gate(concatenated([spectral, spatialPart], axis: -1)))
        let combined = g * spectral + (Float(1) - g) * spatialPart
        return norm(combined).expandedDimensions(axis: 1)
    }

    public func encode(_ input: EncoderInput) throws -> MLXArray {
        guard input.sense == .audio else {
            throw ModelPortError.missingInput(.audio)
        }
        let spectrum = try NeuralOps.batched(input.primary, expected: [AudioEncoder.bins, AudioEncoder.ears],
                                             what: "audio spectrum")
        guard let spatialArray = input.auxiliary else {
            throw ModelPortError.inputShape(what: "audio spatial", expected: [AudioEncoder.spatialNumbers], found: [])
        }
        let spatial = try NeuralOps.batched(spatialArray, expected: [AudioEncoder.spatialNumbers], what: "audio spatial")
        return self(spectrum: spectrum, spatial: spatial)
    }
}
