// AudioTokenizer.swift: the audio tokenizer of the UEMA Apple port (work item WI-30, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-7; contract
// apple/UEMAKit/Specs/audio_render.md sections 3 and 4). Nothing in this file was compiled on the
// machine that wrote it. The "First build on a Mac" checklist of this module is in
// VisionPatcher.swift's header.
//
// INPUT. The audio reader's spectrum for one model step (the last env step's render), float32
// [256, 2]: column 0 the left ear, column 1 the right ear, each bin the summed linear magnitude
// of every sounding source (`p_l`, `p_r` of `CribAudio.render`, Source/Python/mimo_bridge/
// audio_adapter.py:222-269), bins log-spaced from 40 Hz to 8,000 Hz. The render's metadata as
// `auxiliary`, [6] = [left_rms, right_rms, ild_db (dB), itd_ms (ms), azimuth (rad), front_back]
// (audio_adapter.py:285-290; audio_render.md section 3 step 6), or none. This is the one layout
// UEMASenses' AudioReader produces (settled November 4, 2026; the [5] and [2] layouts this
// tokenizer also took before are refused).
//
// OPERATION.
//  1. Spectrum (audio_adapter.py:276-277), in float64, stored float32: db = 10 log10(magnitude +
//     1e-9); value = clip((db + 80) / 80, 0, 1) (`DB_FLOOR` -80, `DB_SPAN` 80, :38). 0 is the
//     -80 dB floor (silence), 1 is 0 dB.
//  2. Spatial pair, the audio encoder's second input: [ild_db / 20, itd_ms / 1.0]
//     (`tokenize_audio`'s metadata[:2], tokenizer/modalities/audio.py:71-73, sliced to 2 at
//     lem_model.py:2476-2477), float32; zeros without metadata (lem_model.py:2472, the `_zm`
//     zeros). The plan row WI-30 names "azimuth / 90"; the encoder reads ILD and ITD, and
//     azimuth / 90 is metadata[2], which no encoder reads: `fullMetadata` gives it for display.
//     The values pass the frame as float32 first (frame_encoder.py:145-166), then the division
//     runs in float64 (audio.py:71-77).
//
// OUTPUT. `primary` float32 [256, 2] on the 0..1 scale, `auxiliary` float32 [2], the audio
// encoder's "spectral" and "spatial" inputs (encoder_registry.py:174, :191).

import Foundation
import UEMACore

/// The audio tokenizer: the two ears' magnitude spectra on the 0..1 log-dB scale plus the
/// spatial pair (audio_adapter.py:276-277; audio.py:71-73; lem_model.py:2464-2479).
public struct AudioTokenizer: Tokenizer, Sendable {
    /// Bins per ear: 256 (`N_BINS`, audio_adapter.py:34; `fft_n_bins`, tokenizer/config.py:258).
    public static let binCount = 256

    /// The dB value mapped to 0: -80 (`DB_FLOOR`, audio_adapter.py:38).
    public static let decibelFloor: Double = -80

    /// The dB range mapped onto 0..1: 80 (`DB_SPAN`, audio_adapter.py:38).
    public static let decibelSpan: Double = 80

    /// Added to every magnitude before the logarithm: 1e-9 (audio_adapter.py:276).
    public static let magnitudeEpsilon: Double = 1e-9

    /// dB per unit of the spatial ILD number: 20 (audio.py:72).
    public static let interauralLevelScaleDecibels: Double = 20

    /// ms per unit of the spatial ITD number: 1 (audio.py:73).
    public static let interauralTimeScaleMilliseconds: Double = 1

    /// Degrees per unit of the metadata azimuth number: 90 (audio.py:74).
    public static let azimuthScaleDegrees: Double = 90

    /// The sense served: audio.
    public var sense: SenseName { .audio }

    /// A tokenizer; it holds no state.
    public init() {}

    /// Step 1 of this file's header for one magnitude.
    public static func unitDecibels(_ magnitude: Float) -> Float {
        let db = 10.0 * log10(Double(magnitude) + magnitudeEpsilon)
        let value = (db - decibelFloor) / decibelSpan
        return Float(min(max(value, 0.0), 1.0))
    }

    /// Step 2 of this file's header: [ild_db / 20, itd_ms / 1.0], each passed through float32 first.
    public static func spatial(interauralLevelDecibels: Float, interauralTimeMilliseconds: Float) -> [Float] {
        [Float(Double(interauralLevelDecibels) / interauralLevelScaleDecibels),
         Float(Double(interauralTimeMilliseconds) / interauralTimeScaleMilliseconds)]
    }

    /// The five metadata numbers of `tokenize_audio` (audio.py:71-77): [ild_db / 20, itd_ms / 1,
    /// azimuth in degrees / 90, front_back, left_rms]; the azimuth is turned from radians to
    /// degrees in float64 (np.degrees multiplies by 180 / pi) and stored float32 first
    /// (frame_encoder.py:162).
    public static func fullMetadata(leftRMS: Float, interauralLevelDecibels: Float, interauralTimeMilliseconds: Float,
                                    azimuthRadians: Float, frontBack: Float) -> [Float] {
        let azimuthDegrees = Float(Double(azimuthRadians) * (180.0 / Double.pi))
        return spatial(interauralLevelDecibels: interauralLevelDecibels,
                       interauralTimeMilliseconds: interauralTimeMilliseconds)
            + [Float(Double(azimuthDegrees) / azimuthScaleDegrees), frontBack, leftRMS]
    }

    /// The encoder input of one reading: `primary` [256, 2], `auxiliary` [2]; `isFresh` carried.
    public func tokenize(_ reading: SenseReading) throws -> EncoderInput {
        try TokenizerChecks.requireSense(reading, .audio)
        guard reading.primary.shape == [AudioTokenizer.binCount, 2] else {
            throw TokenizerError.wrongShape(
                sense: .audio, array: "primary", expected: "[256, 2] linear magnitudes", found: reading.primary.shape)
        }
        let spectrum = reading.primary.values.map(AudioTokenizer.unitDecibels)
        var pair: [Float] = [0, 0]
        if let meta = reading.auxiliary {
            guard meta.shape == [6] else {
                throw TokenizerError.wrongShape(
                    sense: .audio, array: "auxiliary",
                    expected: "[6] (left_rms, right_rms, ild_db, itd_ms, azimuth, front_back)",
                    found: meta.shape)
            }
            pair = AudioTokenizer.spatial(
                interauralLevelDecibels: meta.values[2], interauralTimeMilliseconds: meta.values[3])
        }
        return EncoderInput(
            sense: .audio,
            primary: try ShapedArray(shape: [AudioTokenizer.binCount, 2], values: spectrum),
            auxiliary: try ShapedArray(shape: [2], values: pair),
            isFresh: reading.isFresh)
    }
}
