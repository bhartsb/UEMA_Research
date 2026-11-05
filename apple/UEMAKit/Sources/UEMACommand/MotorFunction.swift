// MotorFunction.swift: the body-function modulation of actuator gain and force range.
//
// UEMAKit module UEMACommand, work item WI-37 (target B) of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html: "the body-function modulation of
// actuator gain and force range (env.py:2840-2845) used by the released run's 'modulate' mode".
// Every file:line below is relative to D:/UEMA_Research/Source/Python. Nothing in this file was
// compiled on the machine that wrote it (Windows 11, no Swift toolchain).
//
// THE PYTHON THIS PORTS (mimo_bridge/env.py:2832-2845, `_apply_motor_function`, read entire on
// November 4, 2026), run after every decode when `body_function` is `modulate` (args.json:51;
// env.py:2892-2893 `if self.body_function_mode == 'modulate': self._apply_motor_function()`):
//   if np.any(self.model.actuator_gaintype!=0) or np.any(self.model.actuator_biastype!=0):
//       raise ValueError('motor-function reference requires validated fixed-gain, bias-free actuators')
//   (first call) _native_actuator_gain = actuator_gainprm.copy();
//                _native_actuator_force_range = actuator_forcerange.copy()
//   factor = ones(nu); known = proprio.act_set >= 0
//   factor[known] = proprio.fatigue.cap[proprio.act_set[known]]
//   actuator_gainprm[:] = native_gain; actuator_gainprm[:, 0] *= factor
//   actuator_forcerange[:] = native_force_range * factor[:, None]
//
// INPUT: the 10 body-set capacities `fatigue.cap`, each 0..1 (1 = fully rested, 0 = drained;
//   proprio_adapter.py:112-116, apple/UEMAKit/Specs/proprio_rows.md section 3), in BODY_SET_ORDER
//   Arm_L, Arm_R, Hand_L, Hand_R, Spine, Head, Leg_L, Leg_R, Foot_L, Foot_R (lem/model/body_sets.py:31-41;
//   proprio_adapter.py:43); the native actuator tables (gainprm column 0, forcerange) read once.
// OPERATION: per actuator a, factor[a] = cap[set of a's joint body] (1.0 when that body is in no
//   set; proprio_adapter.py:170-176 builds act_set); gain0[a] = native_gain0[a] x factor[a];
//   forcerange[a] = native_forcerange[a] x factor[a] (both columns).
// OUTPUT: nu gains and nu (low, high) force ranges, written into mjModel through
//   ActuatorStrengthWriting (the UEMAPhysics adapter). The decoder's control bounds are NOT
//   recomputed: Python fixes them once from the env action space at construction (env.py:1296;
//   actuation.py:136-138), so the modulation lowers the force each control produces, not the
//   control range -1..1.
//
// TIMING. The fatigue model advances once per env step after the physics steps, before the
// sense rows are read (proprio_rows.md section 3; proprio_adapter.py:203-204); the modulation
// runs after the next decode (env.py:2891-2893). So the factors applied before env step t are
// the capacities after env step t-1. MotorCommandWriter in CommandDecoder.swift keeps that order.
//
// FIRST BUILD ON A MAC (with CommandDecoder.swift's checklist):
//  1. `xcodebuild -scheme UEMAKit-Package test -destination 'platform=macOS' -only-testing:UEMACommandTests/MotorFunctionTests`
//     passes every test of Tests/UEMACommandTests/MotorFunctionTests.swift.
//  2. The two adapters at the end of this file join the other modules: ActuatorParameterStrengthWriter
//     over UEMAPhysics's ActuatorParameterWriting (MuJoCoPhysics writes actuator_gainprm[a * 10]
//     and actuator_forcerange[2 a ..< 2 a + 2]), and FatigueCapacities over UEMASenses's
//     ProprioceptionSampling (the shared BodyStateTracker's FatigueModel.capacities, BodySet
//     sequence equal to MotorBodySet's). If WI-25 or WI-27 renamed a member, follow the landed name.
//  3. The tracker must advance its fatigue model once per env step before the next decode
//     (TIMING above); check on the real stack that FatigueCapacities reads the capacities after
//     env step t-1 when env step t is decoded.
//  4. On the WI-15 bundle: MotorFunctionModulator(tables:) does not throw (every actuator has
//     gaintype 0 and biastype 0) and `actuatorBodySet` gives a set to each of the 92 actuators
//     whose joint body is named in body_set_for_mimo_body (proprio_adapter.py:48-68); record
//     the count with no set in apple/FIRST_BUILD_RECORD.md (WI-41).

import Foundation
import UEMAPhysics
import UEMASenses

// MARK: - Body sets

/// The 10 body sets of the model, in BODY_SET_ORDER (lem/model/body_sets.py:31-41: Arm_L 0,
/// Arm_R 1, Hand_L 2, Hand_R 3, Spine 4, Head 5, Leg_L 6, Leg_R 7, Foot_L 8, Foot_R 9). The raw
/// value indexes the capacity array `fatigue.cap`.
public enum MotorBodySet: Int, CaseIterable, Equatable, Sendable {
    case armL = 0
    case armR = 1
    case handL = 2
    case handR = 3
    case spine = 4
    case head = 5
    case legL = 6
    case legR = 7
    case footL = 8
    case footR = 9

    /// The number of body sets: 10 (NUM_BODY_SETS, body_sets.py:44).
    public static let count = 10

    /// Port of `body_set_for_mimo_body` (proprio_adapter.py:46-68): the set a MIMo body
    /// belongs to, or nil for an untracked body (the world, the crib, toys).
    public init?(mimoBodyName name: String) {
        let side: String?
        if name.hasPrefix("left_") {
            side = "L"
        } else if name.hasPrefix("right_") {
            side = "R"
        } else {
            side = nil
        }
        if let side = side {
            // `name.split("_", 1)[1]`: everything after the first underscore.
            let base: String
            if let underscore = name.firstIndex(of: "_") {
                base = String(name[name.index(after: underscore)...])
            } else {
                base = name
            }
            let left = side == "L"
            let handSuffixes = ["distal", "middle", "knuckle", "metacarpal", "thbase", "thhub"]
            if base == "hand" || handSuffixes.contains(where: { base.hasSuffix($0) }) {
                self = left ? .handL : .handR
            } else if base == "upper_arm" || base == "lower_arm" {
                self = left ? .armL : .armR
            } else if base == "upper_leg" || base == "lower_leg" {
                self = left ? .legL : .legR
            } else if base == "foot" || base == "toes" || base == "big_toe" {
                self = left ? .footL : .footR
            } else if base == "eye" {
                self = .head
            } else {
                return nil
            }
            return
        }
        if name == "head" {
            self = .head
        } else if ["hip", "lower_body", "upper_body", "chest"].contains(name) {
            self = .spine
        } else {
            return nil
        }
    }
}

// MARK: - Errors

/// What the modulation refuses, each naming the fact that failed.
public enum MotorFunctionError: Error, Equatable, Sendable {
    /// An actuator's gain type or bias type is not 0 (env.py:2835-2836: "motor-function
    /// reference requires validated fixed-gain, bias-free actuators"): the actuator ids.
    case notFixedGainBiasFree([Int])
    /// The capacity array does not hold 10 numbers (expected, found).
    case capacityCount(expected: Int, found: Int)
}

// MARK: - The modulated parameters and the two injected sides

/// The actuator strength parameters the modulation writes: per actuator, column 0 of
/// actuator_gainprm (the fixed force per unit control) and both columns of actuator_forcerange
/// (in MuJoCo's actuator force units, before the gear: the joint torque is force x gear, and
/// MIMo's gear is the motor's maximum torque in newton-metres, actuation.py:125).
public struct ActuatorStrength: Equatable, Sendable {
    /// actuator_gainprm[:, 0], nu numbers.
    public let gain: [Double]
    /// actuator_forcerange[:, 0], nu numbers.
    public let forceLow: [Double]
    /// actuator_forcerange[:, 1], nu numbers.
    public let forceHigh: [Double]

    /// Makes the parameters; the three arrays are expected to hold nu numbers each.
    public init(gain: [Double], forceLow: [Double], forceHigh: [Double]) {
        self.gain = gain
        self.forceLow = forceLow
        self.forceHigh = forceHigh
    }
}

/// The write of the modulated parameters into the loaded mjModel. The physics type in
/// UEMAPhysics conforms (this file's checklist step 2); tests use a recording fake.
public protocol ActuatorStrengthWriting: AnyObject {
    /// Replaces actuator_gainprm[:, 0] and actuator_forcerange for every actuator.
    func setActuatorStrength(_ strength: ActuatorStrength) throws
}

/// The 10 body-set capacities, each 0..1, in MotorBodySet sequence (`fatigue.cap`,
/// proprio_adapter.py:105-120). The proprioception reader's fatigue state conforms (WI-27).
public protocol BodySetCapacityReading: AnyObject {
    /// `fatigue.cap`, 10 numbers.
    var bodySetCapacities: [Double] { get }
}

// MARK: - The modulator

/// The arithmetic of `_apply_motor_function` (env.py:2832-2845), built once from the native
/// actuator tables: the per-actuator body set and the native gain and force range.
public struct MotorFunctionModulator: Equatable, Sendable {
    /// Per actuator, the body set of its joint's body (`act_set`, proprio_adapter.py:170-176),
    /// nil for an actuator on an untracked body or without a joint transmission.
    public let actuatorBodySet: [MotorBodySet?]
    /// The gain column 0 read once (`_native_actuator_gain[:, 0]`, env.py:2838).
    public let nativeGain: [Double]
    /// The force range low column read once (`_native_actuator_force_range`, env.py:2839).
    public let nativeForceLow: [Double]
    /// The force range high column read once.
    public let nativeForceHigh: [Double]

    /// Reads the native tables. Throws MotorFunctionError.notFixedGainBiasFree when an
    /// actuator has gaintype or biastype other than 0 (env.py:2835-2836).
    public init(tables: ActuatedBodyTables) throws {
        let refused = (0..<tables.actuatorCount).filter {
            tables.actuatorGainType[$0] != ActuatedBodyTables.fixedGainType
                || tables.actuatorBiasType[$0] != ActuatedBodyTables.noBiasType
        }
        if !refused.isEmpty {
            throw MotorFunctionError.notFixedGainBiasFree(refused)
        }
        var sets: [MotorBodySet?] = []
        for a in 0..<tables.actuatorCount {
            // proprio_adapter.py:172-176: j = actuator_trnid[a, 0]; b = jnt_bodyid[j];
            // set = body_set_for_mimo_body(body name).
            guard tables.actuatorTransmissionType[a] == ActuatedBodyTables.jointTransmission else {
                sets.append(nil)
                continue
            }
            let j = tables.actuatorTransmissionID[a]
            let b = tables.jointBodyID[j]
            sets.append(MotorBodySet(mimoBodyName: tables.bodyNames[b]))
        }
        self.actuatorBodySet = sets
        self.nativeGain = tables.actuatorGainParameter0
        self.nativeForceLow = tables.actuatorForceRangeLow
        self.nativeForceHigh = tables.actuatorForceRangeHigh
    }

    /// Per actuator, the factor of env.py:2840-2842: the capacity of its body set, 1.0 for an
    /// actuator of no set. `capacities` must hold 10 numbers.
    public func factors(capacities: [Double]) throws -> [Double] {
        if capacities.count != MotorBodySet.count {
            throw MotorFunctionError.capacityCount(expected: MotorBodySet.count, found: capacities.count)
        }
        return actuatorBodySet.map { set in set.map { capacities[$0.rawValue] } ?? 1.0 }
    }

    /// The modulated gain and force range of env.py:2843-2845.
    public func modulated(capacities: [Double]) throws -> ActuatorStrength {
        let factor = try factors(capacities: capacities)
        var gain = nativeGain
        var low = nativeForceLow
        var high = nativeForceHigh
        for a in 0..<factor.count {
            gain[a] *= factor[a]
            low[a] *= factor[a]
            high[a] *= factor[a]
        }
        return ActuatorStrength(gain: gain, forceLow: low, forceHigh: high)
    }
}

// MARK: - The application, per env step

/// Which body-function mode the run uses (`body_function`, args.json:51 `modulate` on the
/// released run): `modulate` scales gain and force range after every decode; `off` leaves the
/// native tables in place.
public enum BodyFunctionMode: String, Equatable, Sendable {
    case off
    case modulate
}

/// Applies the modulation once per env step, after the decode: reads the capacities from the
/// injected reader and writes the modulated parameters through the injected writer.
public final class MotorFunctionApplier {
    /// The arithmetic.
    public let modulator: MotorFunctionModulator
    /// The mode; `off` makes `apply()` write nothing.
    public let mode: BodyFunctionMode
    private let capacities: BodySetCapacityReading
    private let writer: ActuatorStrengthWriting

    /// The last parameters written, nil before the first `apply()` in `modulate` mode.
    public private(set) var lastWritten: ActuatorStrength?

    /// Makes the applier.
    public init(modulator: MotorFunctionModulator,
                mode: BodyFunctionMode,
                capacities: BodySetCapacityReading,
                writer: ActuatorStrengthWriting) {
        self.modulator = modulator
        self.mode = mode
        self.capacities = capacities
        self.writer = writer
    }

    /// env.py:2892-2893: under `modulate`, the parameters of the current capacities are
    /// written; under `off`, nothing.
    public func apply() throws {
        guard mode == .modulate else {
            return
        }
        let strength = try modulator.modulated(capacities: capacities.bodySetCapacities)
        try writer.setActuatorStrength(strength)
        lastWritten = strength
    }
}

// MARK: - Adapters over UEMAPhysics and UEMASenses

/// ActuatorStrengthWriting over UEMAPhysics's ActuatorParameterWriting: the gain column first,
/// then the force range, the two model writes of env.py:2843-2845.
public final class ActuatorParameterStrengthWriter: ActuatorStrengthWriting {
    private let target: any ActuatorParameterWriting

    /// Wraps the physics that owns mjModel (MuJoCoPhysics on the real scene).
    public init(_ target: any ActuatorParameterWriting) {
        self.target = target
    }

    /// Writes `strength.gain` into actuator_gainprm[:, 0] and the two force-range columns.
    public func setActuatorStrength(_ strength: ActuatorStrength) throws {
        try target.setActuatorGainParameter0(strength.gain)
        try target.setActuatorForceRange(low: strength.forceLow, high: strength.forceHigh)
    }
}

/// BodySetCapacityReading over UEMASenses's ProprioceptionSampling: the 10 capacities of the
/// shared fatigue model (`fatigue.cap`, proprio_adapter.py:105-120), in BodySet sequence, which
/// is BODY_SET_ORDER and MotorBodySet's sequence (body_sets.py:31-41).
public final class FatigueCapacities: BodySetCapacityReading {
    private let source: any ProprioceptionSampling

    /// Wraps the tracker the body sense readers share (WI-27).
    public init(source: any ProprioceptionSampling) {
        self.source = source
    }

    /// The fatigue model's capacities now, each 0..1.
    public var bodySetCapacities: [Double] { source.fatigue.capacities }
}
