// Wellness.swift: the wellness sum of the UEMAModel module (work item WI-36 of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-10 and REQ-B-12 of
// docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html; contract apple/UEMAKit/Specs/wellness_sum.md
// and motor_controller_input.md section 2). Nothing in this file was compiled on the machine that
// wrote it (Windows 11, no Swift toolchain, no Xcode: plan section 3, fact 1).
//
// WHAT THIS FILE DOES.
// INPUT, once per model step (one 50 ms model update):
//   - predicted tokens: the prediction of THIS model step's tokens made on the previous model
//     step (`_nf_pred`), or, on the first model step after a reset, the prediction heads' output
//     of the same model step (lem_model.py:2983-2987, :3008, :3028-3038); 192 numbers per sense;
//   - actual tokens: this model step's raw encoder tokens, 192 numbers per sense; with the set of
//     senses graded "real" this model step (source_mask);
//   - skin sites: every measured tactile patch, 10 normalised fields each (touch_pooler.md
//     section 6: 0 pressure in units of 100 kPa, 3 area in units of 20,000 mm2, 4-7 vibration),
//     2 regional gains (pain, pleasant; 1 and 1 unless a skin profile gives others) and the
//     3-number exposure memory (decaying peak pressure Pa, pressure exposure Pa s, rubbing work
//     J/m2; skin_mechanics.py:129-174) (lem_model.py:2419-2454);
//   - capability: 21 numbers, the first 10 the body-set capacities 0 to 1 (proprio_rows.md
//     section 5);
//   - coherence: the body map's extent, 0 to 1 (wellness.py:237-239; lem/model/body_schema.py,
//     not ported: wellness_sum.md open point A), or nil when no body map runs;
//   - handle state: 26 body handles x 10 numbers (the motor controller's input).
// OPERATION: six wellness channels, each one number 0 to 1 per body, summed (lem_model.py:3099,
//   `total_wellness = torch.cat(list(wellness_out["grounded"].values()), dim=-1).sum()`;
//   `wellness_mode baseline`, args.json:27: wellness.py:504-539 replaces every grounded MLP's
//   output with its formula, so the 26 wellness.* tensors of model.safetensors enter no channel):
//   prediction_error = 1 - clamp(mean over matched senses of the mean over 192 of (predicted -
//     actual)^2, 0, 1); matched = senses with both tokens, graded real, and not vestibular
//     (wellness.py:199-226; `vestibular_wellness_pe exclude`, args.json:176); 1 when none match.
//   tactile_comfort = (0.5 + 0.35 pleasant) (1 - pain) (contact_wellness.py:36-94, profile
//     mechanical_v4, args.json:116); 0.5 when no tactile reading exists (wellness.py:405-406).
//   capability = clamp(mean of capability[0 ..< 10], 0, 1); 0 when absent (wellness.py:277-290).
//   coherence = clamp(extent, 0, 1); without a body map the mean over matched senses of the
//     cosine likeness of predicted to actual, clamped to 0 ... 1 (wellness.py:252-275).
//   motor_state = 1 - mean over 26 x 3 of penalty(centred handle position) with the boundary
//     formula (wellness.py:334-395; `motor_state_formula boundary`, args.json:28).
//   self_touch = 0: its weight `self_touch_wellness` is 0.0 (args.json:155; lem_model.py:8365).
// OUTPUT: the six wellness channels, their sum (ModelOutput.wellness, the number the goal ring
//   stores beside the tokens), the tactile pain (0 to 1, for the harm channel of wellness_sum.md
//   section 4), and the wellness gradient (26 x 9 numbers) the motor controller reads.
//
// THE WELLNESS GRADIENT IN CLOSED FORM. lem_model.py:3101-3106 takes d(total_wellness) /
// d(motor_commands) by autograd, motor_commands = handle_state[:, :, :9] with the three position
// columns minus their mean over the 26 handles (lem_model.py:2999-3001; the centring is outside
// the graph: the derivative is with respect to the centred numbers). Only motor_state reads
// motor_commands (wellness.py:312-318). With rom = rom_limit x 0.21 (BODY_REACH, wellness.py:365;
// rom_limit 1.0 in the mature phase, fixtures' dev_limits), threshold = 0.1 rom, per position
// number p: q = max(|p| - (rom - threshold), 0) / threshold; the forward penalty is min(q, 1)^2
// and the backward slope is that of q^2 for q <= 1 and of 2q - 1 above (the straight-through
// join, wellness.py:385-389). So d motor_state / d p = -(1 / 78) x (q <= 1 ? 2q : 2) x sign(p) /
// threshold where q > 0, and 0 elsewhere; columns 3 to 8 are 0. Checked against torch's autograd
// on a pose with three coordinates past the reach and one inside the boundary band (WellnessTests,
// numbers printed by the torch run quoted there).
//
// WHAT IS NOT IN THIS SUM. The goal ring's raw value `wball` (lem_model.py:4148-4281) adds the
// movement drive's wellness, the agency payment, 0.5 x consequence kinds, the entity self's value
// and 0.25 x the curiosity drive to this sum (wellness_sum.md section 3; open point B there). The
// fixtures record wellness/total as this sum alone (their wellness_channels metadata lists
// "movement" 0.5 beside it, outside the total). Under `wellness_rank normalized` the goal choice
// ranks the normalised channel vector instead (planner_loss.md section 1), which the planner work
// item (WI-39) owns.
//
// MEASURED AGAINST THE FIXTURES (November 4, 2026; the three model-pass fixtures of WI-34). The
// prediction_error channel computed from predicted/<sense> and token/<sense> over vision,
// tactile, audio, proprioception (the motor sense has no head; vestibular excluded) equals
// wellness/prediction_error to 3.1e-8 at most (0.78609705, 0.78870165, 0.79921496); each
// fixture's six channels sum to wellness/total to 9e-8 at most. The tactile_comfort,
// capability and coherence channels cannot be recomputed from the fixtures (they hold the
// 32-point tactile cloud, not the measured patches with their exposure memory; no capability
// vector; no body map), so the fixture test feeds those three channels' recorded values in.
//
// SOLID. S: MotorStateBoundary computes motor_state and its gradient; PredictionErrorChannel,
// TactileComfortChannel, CapabilityChannel and CoherenceChannel one wellness channel each;
// MechanicalLoadPain the local pain of one skin site; GroundedWellnessScorer adds the channels.
// D: GroundedWellnessScorer depends on the LocalPainModel protocol (injected) and is used through
// the WellnessScoring protocol.
//
// FIRST BUILD ON A MAC: MotorController.swift's header, steps 1 to 4 (one command runs both
// test classes). WellnessTests print "wellness <state>: sum <x>, fixture <y>, |difference| <z>"
// per fixture state; the pass line is 1e-5 (the float32 sum of six numbers of at most 1).

import Foundation
import UEMACore

// MARK: - The wellness channels

/// One term of the wellness sum, in the sequence the released model adds them
/// (wellness.py:140-143 EVALUATOR_NAMES, then self_touch, lem_model.py:3097). The raw value is the
/// Python name (the fixtures' wellness/<name> tensors).
public enum WellnessChannel: String, CaseIterable, Codable, Hashable, Sendable {
    case predictionError = "prediction_error"
    case tactileComfort = "tactile_comfort"
    case capability = "capability"
    case coherence = "coherence"
    case motorState = "motor_state"
    case selfTouch = "self_touch"
}

/// Every way the wellness computation refuses.
public enum WellnessError: Error, Equatable, CustomStringConvertible {
    /// An input has `found` numbers where `expected` are required.
    case wrongCount(name: String, expected: Int, found: Int)
    /// An input holds a NaN or an infinity (contact_wellness.py:49-50 refuses the same).
    case nonFinite(String)
    /// A skin reading with zero sites (contact_wellness.py:47-48).
    case noSkinSites
    /// A negative regional gain or exposure memory value (contact_wellness.py:60-61;
    /// skin_mechanics.py:181-182).
    case negative(String)

    public var description: String {
        switch self {
        case .wrongCount(let name, let expected, let found):
            return "\(name) has \(found) numbers; wellness needs \(expected)"
        case .nonFinite(let name):
            return "\(name) holds a NaN or an infinity"
        case .noSkinSites:
            return "a skin reading needs at least one site"
        case .negative(let name):
            return "\(name) is negative"
        }
    }
}

// MARK: - motor_state and the wellness gradient

/// The motor_state wellness channel with the boundary formula, and its derivative with respect
/// to the centred handle pose (this file's header, "THE WELLNESS GRADIENT IN CLOSED FORM").
public struct MotorStateBoundary: Equatable, Sendable {
    /// BODY_REACH: the measured largest handle displacement in position_range units
    /// (wellness.py:365).
    public static let bodyReach = 0.21
    /// The boundary band is the last 10 % of the reach (wellness.py:366).
    public static let thresholdShare = 0.1
    /// Numbers per handle in the handle state (proprio_rows.md section 5).
    public static let handleStateColumns = 10

    /// The developmental rom_limit (1.0 in the mature phase; wellness.py:191, :302-308).
    public let romLimit: Double

    public init(romLimit: Double = 1.0) {
        self.romLimit = romLimit
    }

    /// rom_limit x BODY_REACH, rom_limit floored at 1e-6 (wellness.py:305).
    public var reach: Double { max(romLimit, 1e-6) * MotorStateBoundary.bodyReach }

    /// 0.1 x reach.
    public var threshold: Double { MotorStateBoundary.thresholdShare * reach }

    /// The 234 numbers the formula reads: handle_state[:, :9] with the position columns minus
    /// their mean over the 26 handles (lem_model.py:2999-3001). `handleState` has 26 x 10 numbers.
    public static func centredPose(handleState: [Float]) throws -> [Float] {
        let handles = Command.handleCount
        let columns = handleStateColumns
        guard handleState.count == handles * columns else {
            throw WellnessError.wrongCount(name: "handle state", expected: handles * columns, found: handleState.count)
        }
        guard handleState.allSatisfy({ $0.isFinite }) else {
            throw WellnessError.nonFinite("handle state")
        }
        var means = [Double](repeating: 0, count: 3)
        for handle in 0 ..< handles {
            for axis in 0 ..< 3 {
                means[axis] += Double(handleState[handle * columns + axis])
            }
        }
        for axis in 0 ..< 3 {
            means[axis] /= Double(handles)
        }
        var pose = [Float](repeating: 0, count: Command.count)
        for handle in 0 ..< handles {
            for column in 0 ..< Command.numbersPerHandle {
                let value = Double(handleState[handle * columns + column])
                pose[handle * Command.numbersPerHandle + column] = Float(column < 3 ? value - means[column] : value)
            }
        }
        return pose
    }

    /// q for one position number: the share of the boundary band crossed, 0 inside the
    /// interior, 1 at the reach, above 1 past it.
    private func bandShare(_ position: Double) -> Double {
        let violation = max(threshold - (reach - abs(position)), 0)
        return violation / threshold
    }

    /// 1 - mean over the 26 x 3 position numbers of min(q, 1)^2; 1 when every handle is inside
    /// 90 % of the reach, 0 when every number is at or past it. `centredPose` has 234 numbers.
    public func score(centredPose: [Float]) throws -> Float {
        try MotorStateBoundary.checkPose(centredPose)
        var sum = 0.0
        for handle in 0 ..< Command.handleCount {
            for axis in 0 ..< 3 {
                let q = bandShare(Double(centredPose[handle * Command.numbersPerHandle + axis]))
                let clipped = min(q, 1.0)
                sum += clipped * clipped
            }
        }
        return Float(1.0 - sum / Double(Command.handleCount * 3))
    }

    /// d score / d centredPose, 234 numbers: nonzero only on position numbers with q > 0.
    public func gradient(centredPose: [Float]) throws -> [Float] {
        try MotorStateBoundary.checkPose(centredPose)
        let count = Double(Command.handleCount * 3)
        var out = [Float](repeating: 0, count: Command.count)
        for handle in 0 ..< Command.handleCount {
            for axis in 0 ..< 3 {
                let index = handle * Command.numbersPerHandle + axis
                let position = Double(centredPose[index])
                let q = bandShare(position)
                guard q > 0 else { continue }
                let slope = q <= 1.0 ? 2.0 * q : 2.0
                let sign: Double = position > 0 ? 1.0 : -1.0
                out[index] = Float(-(1.0 / count) * slope * sign / threshold)
            }
        }
        return out
    }

    private static func checkPose(_ pose: [Float]) throws {
        guard pose.count == Command.count else {
            throw WellnessError.wrongCount(name: "centred pose", expected: Command.count, found: pose.count)
        }
        guard pose.allSatisfy({ $0.isFinite }) else {
            throw WellnessError.nonFinite("centred pose")
        }
    }
}

// MARK: - prediction_error and coherence

/// The prediction_error wellness channel (wellness.py:199-226).
public struct PredictionErrorChannel: Equatable, Sendable {
    /// Senses never read even when graded real (`pe_excluded`; vestibular_sense.py:330).
    public let excluded: Set<SenseName>

    public init(excluded: Set<SenseName> = [.vestibular]) {
        self.excluded = excluded
    }

    /// The active senses that enter: present in both sets, graded real, not excluded.
    public func matchedSenses(predicted: TokenSet, actual: TokenSet, realSenses: Set<SenseName>) -> [SenseName] {
        SenseName.active.filter { sense in
            predicted[sense] != nil && actual[sense] != nil && realSenses.contains(sense) && !excluded.contains(sense)
        }
    }

    /// 1 - clamp(mean over matched senses of Token.meanSquaredDifference, 0, 1); 1 when no
    /// sense matches.
    public func score(predicted: TokenSet, actual: TokenSet, realSenses: Set<SenseName>) -> Float {
        let matched = matchedSenses(predicted: predicted, actual: actual, realSenses: realSenses)
        var sum = 0.0
        for sense in matched {
            if let p = predicted[sense], let a = actual[sense] {
                sum += Double(p.meanSquaredDifference(to: a))
            }
        }
        guard !matched.isEmpty else { return 1 }
        let error = sum / Double(matched.count)
        return Float(1.0 - min(max(error, 0), 1))
    }
}

/// Where the coherence channel's number came from.
public enum CoherenceSource: String, Codable, Sendable {
    /// The body map's extent (the released run: `wellness_coherence extent`, args.json:98).
    case bodyMap = "body map extent"
    /// The fallback without a body map: cosine likeness of predicted to actual tokens.
    case predictionLikeness = "prediction likeness (no body map)"
}

/// The coherence wellness channel (wellness.py:228-275).
public struct CoherenceChannel: Equatable, Sendable {
    public let excluded: Set<SenseName>

    public init(excluded: Set<SenseName> = [.vestibular]) {
        self.excluded = excluded
    }

    /// clamp(extent, 0, 1) when the body map gives one; otherwise the mean over the senses
    /// present in both sets (vestibular excluded; the fallback does not read the real grading,
    /// wellness.py:267-268) of cos(predicted, actual), clamped to 0 ... 1; 0 when none.
    public func score(extent: Float?, predicted: TokenSet, actual: TokenSet) -> (value: Float, source: CoherenceSource) {
        if let extent = extent {
            return (min(max(extent, 0), 1), .bodyMap)
        }
        let matched = SenseName.active.filter { predicted[$0] != nil && actual[$0] != nil && !excluded.contains($0) }
        guard !matched.isEmpty else { return (0, .predictionLikeness) }
        var sum = 0.0
        for sense in matched {
            if let p = predicted[sense], let a = actual[sense] {
                sum += CoherenceChannel.cosine(p.values, a.values)
            }
        }
        let mean = sum / Double(matched.count)
        return (Float(min(max(mean, 0), 1)), .predictionLikeness)
    }

    /// torch.nn.functional.cosine_similarity with eps 1e-8: x.y / max(|x| |y|, 1e-8).
    static func cosine(_ x: [Float], _ y: [Float]) -> Double {
        var dot = 0.0
        var xx = 0.0
        var yy = 0.0
        for index in 0 ..< min(x.count, y.count) {
            dot += Double(x[index]) * Double(y[index])
            xx += Double(x[index]) * Double(x[index])
            yy += Double(y[index]) * Double(y[index])
        }
        return dot / max((xx * yy).squareRoot(), 1e-8)
    }
}

// MARK: - capability

/// The capability wellness channel (wellness.py:277-290).
public struct CapabilityChannel: Equatable, Sendable {
    /// The capability vector's length (proprio_rows.md section 5).
    public static let featureCount = 21
    /// How many leading numbers are averaged: the 10 body-set capacities.
    public static let averagedCount = 10

    public init() {}

    /// clamp(mean of features[0 ..< 10], 0, 1); 0 when `features` is nil.
    public func score(features: [Float]?) throws -> Float {
        guard let features = features else { return 0 }
        guard features.count == CapabilityChannel.featureCount else {
            throw WellnessError.wrongCount(name: "capability", expected: CapabilityChannel.featureCount, found: features.count)
        }
        guard features.allSatisfy({ $0.isFinite }) else {
            throw WellnessError.nonFinite("capability")
        }
        var sum = 0.0
        for index in 0 ..< CapabilityChannel.averagedCount {
            sum += Double(features[index])
        }
        return Float(min(max(sum / Double(CapabilityChannel.averagedCount), 0), 1))
    }
}

// MARK: - tactile_comfort

/// One measured skin patch as the tactile_comfort channel reads it (lem_model.py:2419-2454).
public struct SkinSite: Equatable, Sendable {
    /// Fields per patch (touch_pooler.md section 6).
    public static let fieldCount = 10

    /// 10 normalised fields: 0 pressure / 100 kPa, 1 shear, 2 slip, 3 area / 20,000 mm2,
    /// 4-7 vibration power in four bands, 8-9 the remaining fields.
    public let fields: [Float]
    /// Regional gains [pain, pleasant] (contact_wellness.py:41-45); [1, 1] unless a skin profile
    /// gives others.
    public let sensitivity: [Float]
    /// The exposure memory: decaying peak pressure (Pa), pressure exposure (Pa s), rubbing work
    /// per area (J/m2) (skin_mechanics.py:129-134).
    public let history: [Double]
    /// False for a padded patch (contact_wellness.py:53-54, the `valid` mask).
    public let valid: Bool

    public init(fields: [Float], sensitivity: [Float] = [1, 1], history: [Double] = [0, 0, 0], valid: Bool = true) throws {
        guard fields.count == SkinSite.fieldCount else {
            throw WellnessError.wrongCount(name: "skin fields", expected: SkinSite.fieldCount, found: fields.count)
        }
        guard sensitivity.count == 2 else {
            throw WellnessError.wrongCount(name: "skin sensitivity", expected: 2, found: sensitivity.count)
        }
        guard history.count == 3 else {
            throw WellnessError.wrongCount(name: "skin history", expected: 3, found: history.count)
        }
        guard fields.allSatisfy({ $0.isFinite }) else { throw WellnessError.nonFinite("skin fields") }
        guard sensitivity.allSatisfy({ $0.isFinite }) else { throw WellnessError.nonFinite("skin sensitivity") }
        guard history.allSatisfy({ $0.isFinite }) else { throw WellnessError.nonFinite("skin history") }
        guard sensitivity.allSatisfy({ $0 >= 0 }) else { throw WellnessError.negative("skin sensitivity") }
        guard history.allSatisfy({ $0 >= 0 }) else { throw WellnessError.negative("skin history") }
        self.fields = fields
        self.sensitivity = sensitivity
        self.history = history
        self.valid = valid
    }
}

/// The local pain of one skin site, 0 (no modelled aversion) to below 1.
public protocol LocalPainModel {
    func localPain(pressurePascal: Double, history: [Double], painGain: Double) -> Double
}

/// skin_mechanics.mechanical_load with the released profile mechanical_v4 = WIDE_RANGE_PROFILE
/// (contact_wellness.py:73; skin_mechanics.py:87, :177-190): peak = max(pressure, history[0]);
/// acute = max(peak x gain / 219,400 - 1, 0); sustained = history[1] x gain / (219,400 x 60);
/// rubbing = history[2] x gain / 2,194; pain = load / (1 + load), load = acute + sustained +
/// rubbing (the hyperbolic shape, skin_mechanics.py:78).
public struct MechanicalLoadPain: LocalPainModel, Equatable, Sendable {
    /// pressure_reference_pa (skin_mechanics.py:19), the 2-5 year first-discomfort pressure.
    public let pressureReferencePascal: Double
    /// pressure_exposure_seconds (skin_mechanics.py:21).
    public let pressureExposureSeconds: Double
    /// rubbing_reference_j_m2 (skin_mechanics.py:25).
    public let rubbingReferenceJoulesPerSquareMetre: Double

    public init(pressureReferencePascal: Double = 219_400, pressureExposureSeconds: Double = 60,
                rubbingReferenceJoulesPerSquareMetre: Double = 2_194) {
        self.pressureReferencePascal = pressureReferencePascal
        self.pressureExposureSeconds = pressureExposureSeconds
        self.rubbingReferenceJoulesPerSquareMetre = rubbingReferenceJoulesPerSquareMetre
    }

    public func localPain(pressurePascal: Double, history: [Double], painGain: Double) -> Double {
        let peak = max(pressurePascal, history[0])
        let acute = max(peak * painGain / pressureReferencePascal - 1, 0)
        let sustained = history[1] * painGain / (pressureReferencePascal * pressureExposureSeconds)
        let rubbing = history[2] * painGain / rubbingReferenceJoulesPerSquareMetre
        let load = max(acute + sustained + rubbing, 0)
        return load / (1 + load)
    }
}

/// What the tactile_comfort channel computed.
public struct TactileComfortResult: Equatable, Sendable {
    /// (0.5 + 0.35 pleasant) (1 - pain), 0 to 0.85.
    public let comfort: Float
    /// The largest local pain over the sites, 0 to below 1; the pain channel of
    /// wellness_sum.md section 4.
    public let pain: Float
    /// The area-weighted mean of the active sites' pleasantness, 0 to 1.
    public let pleasant: Float
}

/// The tactile_comfort wellness channel (contact_wellness.py:36-94).
public struct TactileComfortChannel {
    /// TACTILE_PLEASANT_PEAK, pressure units (contact_wellness.py:17): 0.05 = 5 kPa.
    public static let pleasantPeak = 0.05
    /// TACTILE_PEAK_WIDTH in decades of pressure (contact_wellness.py:18).
    public static let peakWidth = 0.6
    /// TACTILE_TEXTURE_REF (contact_wellness.py:33).
    public static let textureReference = 0.5
    /// A site is active above this pressure (contact_wellness.py:52).
    public static let activePressure = 1e-6
    /// Pascal per pressure unit (contact_wellness.py:74: pressure * 100000.).
    public static let pascalPerUnit = 100_000.0

    private let painModel: any LocalPainModel

    public init(painModel: any LocalPainModel = MechanicalLoadPain()) {
        self.painModel = painModel
    }

    /// `sites` nil means no tactile reading this model step: comfort 0.5, pain 0, pleasant 0
    /// (wellness.py:405-406). An empty array throws (contact_wellness.py:47-48).
    public func evaluate(sites: [SkinSite]?) throws -> TactileComfortResult {
        guard let sites = sites else {
            return TactileComfortResult(comfort: 0.5, pain: 0, pleasant: 0)
        }
        guard !sites.isEmpty else { throw WellnessError.noSkinSites }
        let c = TactileComfortChannel.self
        var pain = 0.0
        var weightedPleasant = 0.0
        var weightSum = 0.0
        for site in sites {
            let pressure = max(Double(site.fields[0]), 0)
            let active = pressure > c.activePressure && site.valid
            var localPain = painModel.localPain(pressurePascal: pressure * c.pascalPerUnit,
                                                history: site.history, painGain: Double(site.sensitivity[0]))
            if !site.valid {
                localPain = 0
            }
            pain = max(pain, localPain)
            guard active else { continue }
            let decades = log10(max(pressure, 1e-5)) - log10(c.pleasantPeak)
            let gentle = exp(-(decades * decades) / (2 * c.peakWidth * c.peakWidth))
            var vibration = 0.0
            for band in 4 ..< 8 {
                vibration += abs(Double(site.fields[band]))
            }
            let texture = tanh((vibration / 4) / c.textureReference)
            let localPleasant = min(max(gentle * (0.5 + 0.5 * texture) * Double(site.sensitivity[1]), 0), 1)
            let area = max(Double(site.fields[3]), 0)
            let weight = area <= 0 ? 1.0 : area
            weightedPleasant += localPleasant * weight
            weightSum += weight
        }
        let pleasant = weightedPleasant / max(weightSum, 1e-12)
        let comfort = (0.5 + 0.35 * pleasant) * (1 - pain)
        return TactileComfortResult(comfort: Float(comfort), pain: Float(pain), pleasant: Float(pleasant))
    }
}

// MARK: - The sum

/// Everything one model step's wellness reads (this file's header, INPUT).
public struct WellnessInputs: Sendable {
    /// The prediction of this model step's tokens (see the header).
    public let predicted: TokenSet
    /// This model step's raw encoder tokens.
    public let actual: TokenSet
    /// The senses graded "real" this model step.
    public let realSenses: Set<SenseName>
    /// Every measured skin patch; nil when the tactile sense gave no reading.
    public let skin: [SkinSite]?
    /// 21 capability numbers; nil when absent.
    public let capability: [Float]?
    /// The body map's extent 0 to 1; nil when no body map runs.
    public let coherenceExtent: Float?
    /// 26 body handles x 10 numbers.
    public let handleState: [Float]

    public init(predicted: TokenSet, actual: TokenSet, realSenses: Set<SenseName>, skin: [SkinSite]?,
                capability: [Float]?, coherenceExtent: Float?, handleState: [Float]) {
        self.predicted = predicted
        self.actual = actual
        self.realSenses = realSenses
        self.skin = skin
        self.capability = capability
        self.coherenceExtent = coherenceExtent
        self.handleState = handleState
    }
}

/// One model step's wellness.
public struct WellnessScore: Equatable, Sendable {
    /// Each wellness channel's number, 0 to 1.
    public let channels: [WellnessChannel: Float]
    /// The sum of the six wellness channels in WellnessChannel's sequence, 0 to 6.
    public let total: Float
    /// d total / d centred handle pose, 26 x 9 numbers: the motor controller's input.
    public let gradient: [Float]
    /// The tactile channel's pain, 0 to below 1.
    public let tactilePain: Float
    /// Where coherence came from.
    public let coherenceSource: CoherenceSource

    /// The wellness channels' numbers in WellnessChannel's sequence.
    public var orderedChannels: [Float] {
        WellnessChannel.allCases.map { channels[$0] ?? 0 }
    }
}

/// Scores one model step; the forward model (WI-35) and the session (WI-32) depend on this
/// protocol.
public protocol WellnessScoring {
    func score(_ inputs: WellnessInputs) throws -> WellnessScore
}

/// The released wellness sum (this file's header, OPERATION).
public struct GroundedWellnessScorer: WellnessScoring {
    /// `self_touch_wellness` (args.json:155): the self_touch channel's weight; 0 makes the
    /// channel 0 on every model step (lem_model.py:8365).
    public static let selfTouchWeight: Float = 0

    public let motorState: MotorStateBoundary
    public let predictionError: PredictionErrorChannel
    public let coherence: CoherenceChannel
    public let capability: CapabilityChannel
    private let tactile: TactileComfortChannel

    public init(motorState: MotorStateBoundary = MotorStateBoundary(),
                predictionError: PredictionErrorChannel = PredictionErrorChannel(),
                coherence: CoherenceChannel = CoherenceChannel(),
                capability: CapabilityChannel = CapabilityChannel(),
                painModel: any LocalPainModel = MechanicalLoadPain()) {
        self.motorState = motorState
        self.predictionError = predictionError
        self.coherence = coherence
        self.capability = capability
        self.tactile = TactileComfortChannel(painModel: painModel)
    }

    public func score(_ inputs: WellnessInputs) throws -> WellnessScore {
        let pose = try MotorStateBoundary.centredPose(handleState: inputs.handleState)
        let tactileResult = try tactile.evaluate(sites: inputs.skin)
        let coherenceResult = coherence.score(extent: inputs.coherenceExtent,
                                              predicted: inputs.predicted, actual: inputs.actual)
        let predictionValue = predictionError.score(predicted: inputs.predicted, actual: inputs.actual,
                                                    realSenses: inputs.realSenses)
        let capabilityValue = try capability.score(features: inputs.capability)
        let motorValue = try motorState.score(centredPose: pose)
        let gradient = try motorState.gradient(centredPose: pose)
        // self_touch: arrived x weight x gate (lem_model.py:8365) with weight 0 is 0.
        let selfTouchValue: Float = 0
        let channels: [WellnessChannel: Float] = [
            .predictionError: predictionValue,
            .tactileComfort: tactileResult.comfort,
            .capability: capabilityValue,
            .coherence: coherenceResult.value,
            .motorState: motorValue,
            .selfTouch: selfTouchValue,
        ]
        return WellnessScore(channels: channels,
                             total: GroundedWellnessScorer.sum(channels),
                             gradient: gradient,
                             tactilePain: tactileResult.pain,
                             coherenceSource: coherenceResult.source)
    }

    /// The float32 sum of the channels in WellnessChannel's sequence (lem_model.py:3099 adds the
    /// dict's values in that sequence).
    public static func sum(_ channels: [WellnessChannel: Float]) -> Float {
        var total: Float = 0
        for channel in WellnessChannel.allCases {
            total += channels[channel] ?? 0
        }
        return total
    }
}

// MARK: - The wellness tensors the sum does not read

/// The 26 tensors of the manifest's wellness component (295,914 parameters, receipt
/// component_params.wellness) with their shapes. None enters the sum: the five grounded MLPs'
/// outputs are replaced by the formulas (wellness.py:504-539) and the approximate evaluator
/// (Linear(960, 256), GELU, Linear(256, 128), GELU, Linear(128, 5), Sigmoid; wellness.py:73-91)
/// reaches no goal choice and no command with `prospective_wellness off` (wellness_sum.md
/// section 5). The loader still loads them (REQ-B-11); this table lets a test hold the port's
/// reading of the component equal to the manifest.
public enum WellnessTensorTable {
    /// Grounded MLP input widths (master_config.yaml:329-344; motor_state 234 on this body,
    /// body_layout.py:54) and hidden widths.
    public static let groundedLayers: [(name: String, input: Int, hidden: Int)] = [
        (name: "capability", input: 21, hidden: 32),
        (name: "coherence", input: 1, hidden: 32),
        (name: "motor_state", input: 234, hidden: 64),
        (name: "prediction_error", input: 1, hidden: 32),
        (name: "tactile_comfort", input: 8, hidden: 32),
    ]

    /// Every key with its shape, sorted by key (the manifest's sequence).
    public static let tensorShapes: [(key: String, shape: [Int])] = {
        var out: [(key: String, shape: [Int])] = [
            (key: "wellness.approximate.net.0.bias", shape: [256]),
            (key: "wellness.approximate.net.0.weight", shape: [256, 960]),
            (key: "wellness.approximate.net.2.bias", shape: [128]),
            (key: "wellness.approximate.net.2.weight", shape: [128, 256]),
            (key: "wellness.approximate.net.4.bias", shape: [5]),
            (key: "wellness.approximate.net.4.weight", shape: [5, 128]),
        ]
        for layer in WellnessTensorTable.groundedLayers {
            let base = "wellness.grounded." + layer.name + ".net."
            out.append((key: base + "0.bias", shape: [layer.hidden]))
            out.append((key: base + "0.weight", shape: [layer.hidden, layer.input]))
            out.append((key: base + "2.bias", shape: [1]))
            out.append((key: base + "2.weight", shape: [1, layer.hidden]))
        }
        return out.sorted { $0.key < $1.key }
    }()
}
