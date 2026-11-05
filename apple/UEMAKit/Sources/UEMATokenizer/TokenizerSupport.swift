// TokenizerSupport.swift: the error type and the shape checks the six tokenizers of the
// UEMATokenizer module share (work item WI-30, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; plan section 2, stage 3).
//
// Nothing in this module was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). Every file of the module imports Foundation and
// UEMACore only: no MLX symbol is used, because the contract type `EncoderInput`
// (UEMACore/Protocols.swift) holds plain float32 arrays, and no UEMASenses symbol is used,
// so that a fake `SenseReading` in a test needs nothing but UEMACore.
//
// THE MODULE (one tokenizer per active sense, each a `Tokenizer` of UEMACore):
//   VisionPatcher.swift            vision: 16 x 16 x 3 patches / 255 -> [1160, 768] + [1160, 4]
//   TactileTokenizer.swift         tactile: [<=256, 10] fields / _FIELD_NORMS -> [32, 19]
//   AudioTokenizer.swift           audio: [256, 2] magnitude -> 0..1 log-dB [256, 2] + [2]
//   ProprioceptionTokenizer.swift  proprioception: [26, 11] rows -> [312]
//   MotorTokenizer.swift           motor: [26, 10] achieved rows -> [26, 19]
//   VestibularNormaliser.swift     vestibular: [29] fields -> [28] centred and clipped
//   TokenizerBank.swift            the six together, keyed by sense
// FIRST BUILD ON A MAC: the checklist is in VisionPatcher.swift's header.

import Foundation
import UEMACore

/// What a tokenizer refuses: a reading of another sense, an array of a shape the sense does
/// not have, a missing camera image, or statistics of the wrong length.
public enum TokenizerError: Error, Equatable, Sendable {
    /// The reading belongs to `found`; this tokenizer serves `expected`.
    case wrongSense(expected: SenseName, found: SenseName)
    /// `array` of a `sense` reading has the shape `found`; `expected` says which shapes are
    /// accepted, in words.
    case wrongShape(sense: SenseName, array: String, expected: String, found: [Int])
    /// The camera `camera` has no image in the set given to the vision patcher.
    case missingImage(camera: String)
    /// The image of `camera` is `width` x `height` pixels; the patcher's source needs
    /// `expectedSize` x `expectedSize` (any size whose 16-pixel grid is the same is accepted,
    /// as `image_to_patches` crops to a multiple of 16, vision_adapter.py:110-112).
    case imageSizeMismatch(camera: String, expectedSize: Int, width: Int, height: Int)
    /// A vision source setting outside its range: `reason` names it.
    case invalidSource(reason: String)
    /// Normaliser statistics with `found` numbers where `expected` are required.
    case wrongStatisticsCount(name: String, expected: Int, found: Int)
    /// Two tokenizers given to one `TokenizerBank` serve this sense.
    case duplicateSense(SenseName)
}

/// Shape checks shared by the tokenizers.
enum TokenizerChecks {
    /// Throws `TokenizerError.wrongSense` unless `reading.sense` is `sense`.
    static func requireSense(_ reading: SenseReading, _ sense: SenseName) throws {
        guard reading.sense == sense else {
            throw TokenizerError.wrongSense(expected: sense, found: reading.sense)
        }
    }

    /// The number of rows of a rank-2 array whose column count is in `columns`; throws
    /// `TokenizerError.wrongShape` otherwise.
    static func rows(of array: ShapedArray, columns: Set<Int>, sense: SenseName, name: String,
                     expected: String) throws -> Int {
        guard array.rank == 2, columns.contains(array.shape[1]) else {
            throw TokenizerError.wrongShape(sense: sense, array: name, expected: expected, found: array.shape)
        }
        return array.shape[0]
    }
}
