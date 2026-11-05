// TokenizerBank.swift: the six tokenizers of one session keyed by sense (work item WI-30, target B
// of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; plan section 2, stage 3, "six
// tokenizers -> encoder inputs"). Nothing in this file was compiled on the machine that wrote
// it. The "First build on a Mac" checklist of this module is in VisionPatcher.swift's header.
//
// INPUT. Any tokenizers (one per sense, injected at init), then per model step the readings of
// the senses. OPERATION. Each reading goes to the tokenizer of its sense. OUTPUT. One
// `EncoderInput` per reading, keyed by sense. The session (RunViewModel, plan row WI-32) may hold
// a bank or the six tokenizers directly; the bank adds nothing but the lookup.

import Foundation
import UEMACore

/// One tokenizer per sense, injected; tokenizes a model step's readings by sense.
public struct TokenizerBank {
    /// The tokenizers by the sense each serves.
    public let tokenizers: [SenseName: any Tokenizer]

    /// A bank over `tokenizers`; two for one sense throw `TokenizerError.duplicateSense`.
    public init(_ tokenizers: [any Tokenizer]) throws {
        var bySense: [SenseName: any Tokenizer] = [:]
        for tokenizer in tokenizers {
            if bySense[tokenizer.sense] != nil {
                throw TokenizerError.duplicateSense(tokenizer.sense)
            }
            bySense[tokenizer.sense] = tokenizer
        }
        self.tokenizers = bySense
    }

    /// The released bank: the six active senses' tokenizers, with `vestibular` the checkpoint's
    /// frozen normaliser.
    public static func released(vestibular: VestibularNormaliser) -> TokenizerBank {
        let list: [any Tokenizer] = [
            VisionPatcher.released(), TactileTokenizer(), AudioTokenizer(), ProprioceptionTokenizer(),
            MotorTokenizer(), vestibular,
        ]
        var bySense: [SenseName: any Tokenizer] = [:]
        for tokenizer in list {
            bySense[tokenizer.sense] = tokenizer
        }
        return TokenizerBank(bySense: bySense)
    }

    private init(bySense: [SenseName: any Tokenizer]) {
        self.tokenizers = bySense
    }

    /// The encoder inputs of `readings`, keyed by sense. A reading of a sense without a
    /// tokenizer throws `UEMACoreError.missingSense`; a later reading of the same sense replaces
    /// an earlier one.
    public func tokenize(_ readings: [SenseReading]) throws -> [SenseName: EncoderInput] {
        var out: [SenseName: EncoderInput] = [:]
        for reading in readings {
            guard let tokenizer = tokenizers[reading.sense] else {
                throw UEMACoreError.missingSense(reading.sense)
            }
            out[reading.sense] = try tokenizer.tokenize(reading)
        }
        return out
    }
}
