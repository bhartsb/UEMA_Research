// SenseName.swift: the seven senses of the model registry, the six active in the released
// run, and the token (192 numbers) each active sense produces per model step.
// Work item WI-20, target B of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). Plain Swift: no import beyond the standard library.
// The "First build on a Mac" checklist of this work item is in Protocols.swift's header.
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave:
// - weights/b1_continue_B1K_skip13_492800/run/trainer.log line 5: "LemLite: 18131745 params,
//   active=['vision', 'tactile', 'audio', 'proprioception', 'motor', 'vestibular'],
//   d_model=192, ctx=128, amp=True, imagination=True, wellness=baseline": the six active
//   senses in this sequence; a token is d_model = 192 numbers; the context is 128 model steps.
// - Source/Python/lem/model/encoder_registry.py line 33: "returns a dict of {modality_name:
//   [B, 1, d_model]} tensors.": one token per active sense per model step.
// - README.md lines 17-18: "Each of seven senses (vision, tactile, audio, proprioception,
//   motor, vestibular, intermind) is encoded into tokens by its own encoder."
// - docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html, REQ-B-8: "Intermind. Not in the
//   active list; the Python fills a synthetic placeholder the transformer never takes as a
//   token [I2 intermind; I1 receipt]; UEMAKit has none."; its pass line: "The forward
//   model's input shape is [768,192]." (6 senses x 128 model steps = 768 tokens).
// - apple/UEMAKit/Specs/vestibular_fields.md section 1: "`vestibular_sense on`
//   (args.json:174) puts this sense in the sixth slot in place of intermind
//   (vestibular_sense.py:120-121, 337-342)."
// - apple/UEMAKit/Specs/motor_controller_input.md section 1: the motor controller's pooled
//   input is "one 192-number vector per sense in the sequence vision, tactile, audio,
//   proprioception, motor, vestibular (receipt `active_modalities`), laid end to end".
// - apple/UEMAKit/Specs/planner_loss.md section 3 step 5: the planner's goal term sums
//   "over m in {vision, tactile, audio, proprioception, vestibular}" ("motor excluded").
// - Source/Python/lem/model/encoder_registry.py line 53: "predictive_state.py documents the
//   invariant that a real encoder token has RMS 1.0000."
// - The per-sense input shapes in the case comments: specification REQ-B-4 (proprioception
//   [26,12] flattened to 312, vestibular [28], motor [26,19]), REQ-B-5 (tactile [32,19]),
//   REQ-B-6 (vision patches [N,768] with angles [N,4]), REQ-B-7 (audio [256,2] with [2]).

// MARK: - The senses

/// One of the seven senses of the model registry (README.md:17-18), by the registry's own
/// name. Six are active in the released run (trainer.log line 5); `intermind` is inactive:
/// the Python trainer fills a placeholder for it that the forward model never takes as a
/// token (specification REQ-B-8), and UEMAKit computes nothing for it. The raw value is the
/// registry's name, as the weights manifest and the fixtures spell it.
public enum SenseName: String, CaseIterable, Codable, Hashable, Sendable {
    /// Five camera images per render (eye_left and eye_right 256 px over 60 degrees,
    /// fovea_left and fovea_right 288 px over 24 degrees, ring_left 256 px over 150 degrees),
    /// cut into 16 x 16 x 3 patches of 768 numbers with 4 angle numbers each (REQ-B-6; the
    /// manifest row encoder_registry.encoders.vision.angular_pos_enc.mlp.0.weight has shape
    /// [192, 4], apple/UEMAKit/Sources/UEMAModel/Resources/weight_manifest.json).
    case vision
    /// Contact forces over the 4,683 skin points, pooled to 256 patches x 10 fields and
    /// sorted by pressure into [32, 19] per body (REQ-B-5).
    case tactile
    /// 256 spectrum bins per ear on a 0 to 1 scale ([256, 2]) plus 2 spatial numbers (REQ-B-7):
    /// the first 2 of the tokenizer's 5 metadata numbers, interaural level difference / 20 dB
    /// and interaural time difference / 1.0 ms (audio_render.md section 4; the `[:2]` cut at
    /// mimo_bridge/lem_model.py:2476-2477; manifest row
    /// encoder_registry.encoders.audio.spatial_proj.weight shape [192, 2]).
    case audio
    /// 26 handle rows x 12 numbers (orientation, position, velocity, fatigue capacity, row
    /// index), flattened to 312 (REQ-B-4).
    case proprioception
    /// 26 handle rows x 19 numbers: the achieved handle state (10) and the executed command
    /// (9) (REQ-B-4; apple/UEMAKit/Specs/proprio_rows.md section 5).
    case motor
    /// 28 head-motion, neck, support and effort fields, centred and clipped (REQ-B-4;
    /// apple/UEMAKit/Specs/vestibular_fields.md).
    case vestibular
    /// Inactive in the released run: trainer.log line 5 lists six active senses without it,
    /// and vestibular took its sixth slot (vestibular_fields.md section 1).
    case intermind

    /// The six active senses in the sequence the released run lists them (trainer.log
    /// line 5). This is the sequence of the six tokens inside one model step of the
    /// forward model's 768-token context and of the motor controller's pooled input
    /// (motor_controller_input.md section 1).
    public static let active: [SenseName] = [
        .vision, .tactile, .audio, .proprioception, .motor, .vestibular,
    ]

    /// The number of active senses: 6.
    public static let activeCount = 6

    /// The five senses whose predicted token enters the planner's goal term; motor is
    /// excluded (planner_loss.md section 3 step 5).
    public static let goalScored: [SenseName] = [
        .vision, .tactile, .audio, .proprioception, .vestibular,
    ]

    /// True for the six senses the released run encodes; false for intermind.
    public var isActive: Bool { self != .intermind }

    /// The position of this sense among the six tokens of one model step, 0 (vision) to 5
    /// (vestibular); nil for intermind, which has no token.
    public var tokenIndex: Int? { SenseName.active.firstIndex(of: self) }
}

// MARK: - The token

/// One token: the 192 float32 numbers an encoder produces for one active sense on one
/// model step (`d_model` 192, trainer.log line 5; encoder_registry.py line 33). The
/// prediction heads' output per sense and every goal ring entry have the same shape. A real
/// encoder token has a root mean square of 1.0 after the registry's shared output LayerNorm
/// (encoder_registry.py line 53); `Token.zero` is a placeholder, never an encoder's output.
public struct Token: Equatable, Hashable, Codable, Sendable {
    /// The numbers per token: 192.
    public static let count = 192

    /// The 192 numbers.
    public let values: [Float]

    /// Wraps exactly 192 numbers; any other count throws `UEMACoreError.wrongCount`.
    public init(values: [Float]) throws {
        guard values.count == Token.count else {
            throw UEMACoreError.wrongCount(type: "Token", expected: Token.count, found: values.count)
        }
        self.values = values
    }

    /// A token whose 192 numbers all equal `value`.
    public init(repeating value: Float) {
        values = Array(repeating: value, count: Token.count)
    }

    /// The all-zero token.
    public static let zero = Token(repeating: 0)

    /// The number at `index`, 0 to 191.
    public subscript(index: Int) -> Float { values[index] }

    /// The mean over the 192 positions of the squared difference to `other`: the planner's
    /// per-sense goal term (planner_loss.md section 3 step 5) and the sense panel's prediction
    /// error (specification REQ-B-14). 0 when equal.
    public func meanSquaredDifference(to other: Token) -> Float {
        var sum: Float = 0
        for index in 0 ..< Token.count {
            let difference = values[index] - other.values[index]
            sum += difference * difference
        }
        return sum / Float(Token.count)
    }

    /// The square root of the mean of the 192 squared numbers; 1.0 for a real encoder token.
    public var rootMeanSquare: Float {
        var sum: Float = 0
        for value in values {
            sum += value * value
        }
        return (sum / Float(Token.count)).squareRoot()
    }

    private enum CodingKeys: String, CodingKey {
        case values
    }

    /// Decodes through the count check, so a decoded token also holds 192 numbers.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(values: container.decode([Float].self, forKey: .values))
    }
}

// MARK: - A set of tokens keyed by sense

/// The tokens of one model step keyed by sense: six entries when every active sense has one.
public typealias TokenSet = [SenseName: Token]

extension Dictionary where Key == SenseName, Value == Token {
    /// The active senses that have no token in this set, in the active sequence; empty when
    /// the set is complete.
    public var missingActiveSenses: [SenseName] {
        SenseName.active.filter { self[$0] == nil }
    }

    /// The six tokens in the active sequence (vision first, vestibular last); throws
    /// `UEMACoreError.missingSense` naming the first active sense without a token.
    public func activeSequence() throws -> [Token] {
        try SenseName.active.map { sense in
            guard let token = self[sense] else {
                throw UEMACoreError.missingSense(sense)
            }
            return token
        }
    }
}
