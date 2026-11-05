// Planner.swift: one planning event. Adam on one command (26 body handles x 9 numbers =
// 234 numbers) through the frozen forward model, toward the goal ring's goal entry, scoring
// the start and every update and keeping the best-scored command per body. Work item WI-39,
// target B of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html (specification
// REQ-B-12; the specification's section 5 names the concrete type AdamPlanner).
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1).
//
// INPUT, OPERATION, OUTPUT (apple/UEMAKit/Specs/planner_loss.md sections 1 to 3; Python
// paths under D:/UEMA_Research/Source/Python):
// - Input: the motor controller's command of this model step (234 numbers, positions in
//   position_range units = cm / 200); the goal entry's tokens (192 numbers per sense); a
//   `PlanTransition`, the differentiable map from a raw command [B, 26, 9] to the predicted
//   next tokens [B, 192] per sense through the frozen forward model and prediction heads of
//   this model step (predictive_state.py:148-256, called at intention_generator.py:331-334).
// - Parameters: position [B, 26, 3], rotation [B, 26, 4], radius logit [B, 26, 1], strength
//   logit [B, 26, 1], each logit = log(v / (1 - v)) of the value clamped to 1e-4 ... 1 - 1e-4
//   (intention_generator.py:244-252). The command assembled from them: position, rotation
//   divided by max(its norm, 1e-12), sigmoid(radius logit), sigmoid(strength logit)
//   (intention_generator.py:296-300).
// - Score per body b: goal[b] = sum over the scored senses (vision, tactile, audio,
//   proprioception, vestibular; motor excluded; a sense whose source is not "real" left out)
//   of the mean over 192 numbers of (predicted - goal)^2 (intention_generator.py:343-352);
//   effort[b] = 0.01 x the sum over the 26 handles of strength^2 (:355, effort per body);
//   total[b] = goal[b] + effort[b] (:370; the extra and wellness terms are 0 in the released
//   run, planner_loss.md section 3 step 7).
// - Loop: evaluations 0 to 20 (21); after evaluations 0 to 19, the gradient of the mean of
//   total over bodies with respect to the four parameter arrays and one Adam update at
//   learning rate 0.1 (intention_generator.py:301, 392-399); a body's selected evaluation
//   changes only when its total is finite and strictly below its best so far (:378-385); a
//   non-finite start throws, a later non-finite score or gradient stops the loop (:373-375,
//   :389-397).
// - Output: the selected raw command per body (before the execution transform,
//   planner_loss.md section 3 end), the 21 scores (total, goal, effort) per body, the
//   selected evaluation per body, the update count and the stop reason.
//
// ADAM IS WRITTEN OUT HERE. The search updates its parameters with `TorchAdam`, the update
// of torch.optim.Adam (bias-corrected first and second moments, eps added after the square
// root of the corrected second moment), because the Python planner runs torch.optim.Adam
// (intention_generator.py:271) and the fixtures of WI-34 were recorded with it. MLX's Adam in
// MLXOptimizers is the other choice the specification names; MLX's Python Adam takes
// `bias_correction=False` by default (recalled from the MLX source, not fetched in this
// item; whether the Swift Adam matches it is unchecked), and without bias correction the
// first update moves a parameter by 0.1 x (1 - 0.9) / sqrt(1 - 0.999) = 0.316 instead of
// 0.1. MLX supplies the gradient (valueAndGrad); the 234 parameters and their moments live
// in Swift arrays.
//
// FIRST BUILD ON A MAC (this work item's acceptance: "on an exact scalar forward model
// (test_planner_selection.py's pattern) the planner returns the command it scored best; with
// the real model, 21 evaluations counted and the chosen score at or below the start"):
//  1. Run WI-17's steps 1 to 4 (XCFramework in place, `swift package resolve` exit 0,
//     `xcodebuild -scheme UEMAKit-Package build -destination 'platform=macOS'` exit 0).
//  2. `xcodebuild -scheme UEMAKit-Package test -destination 'platform=macOS'
//     -only-testing:UEMAPlannerTests` : exit 0. The test files: GoalRingTests.swift,
//     AdamCommandSearchTests.swift (classes TorchAdamTests and AdamCommandSearchTests),
//     PlannerTests.swift, HandDescentTests.swift, the shared PlannerTestSupport.swift, and the
//     WI-17 placeholder UEMAPlannerPackageTests.swift.
//  3. Constructs to check first if step 1 or 2 fails, each used on purpose:
//     `valueAndGrad(_:argumentNumbers:)` over `[MLXArray] -> [MLXArray]` with the loss as
//     the first output and four auxiliary outputs after it (if the build's gradient
//     includes the auxiliary outputs, `testGradientStepMatchesAdamOnTheScalarModel` fails:
//     then differentiate a closure returning only the loss and score with a second forward
//     call); `MLXArray(_:_:)` from [Float] and a shape; `sum(axis:keepDims:)`,
//     `mean(axis:)`, `mean()`, `reshaped(_:)`, `concatenated(_:axis:)`, `MLX.sigmoid`,
//     `MLX.maximum`, `MLX.minimum`, `MLX.sqrt`, `MLX.matmul` (tests), `which(_:_:_:)` with a
//     Bool condition array `MLXArray([Bool], shape)` (HandDescent.swift), `asArray(Float.self)`.
//     If `testInvalidBodyWithANonFiniteCostDoesNotThrow` fails, `which` did not mask the NaN
//     cost of the invalid body (hand_response.py:171 masks it to 0).
//  4. The real-model half of the acceptance runs after WI-35 and WI-36: a `PlanTransition`
//     over UEMAModel's forward model (open point below) with a WI-34 fixture state; check
//     `CommandSearchResult.evaluations.count == 21` and `selectedTotal(body: 0) <=
//     evaluations[0][0].total`, and compare the 21 totals with the fixture's within the
//     tolerance recorded in plan section 9.
//  5. Record each step's command, exit code and clock time (AM/PM Pacific) in
//     apple/FIRST_BUILD_RECORD.md (WI-41).
//
// OPEN POINTS (each one named in this work item's result):
// - The concrete `PlanTransition` over WI-35's forward model, including the execution
//   transform (positions limited to a 40 cm step from the achieved handle, the 0.35 blend
//   with the previous proposal) and the division by 200 (planner_loss.md section 3 steps 2
//   to 4). UEMAModel's forward model was not on disk when this file was written.
// - The released run plans in `HandActionSpace` (planner_loss.md section 2: 44 joint
//   parameters beside the four arrays, radius clamped to 0 ... 1 rather than a sigmoid, the
//   finger rows rebuilt from the joint goal by the hand kinematics). This file ports the
//   direct action space (`action_space is None`, intention_generator.py:296-300) as
//   `DirectCommandParameterization`; the hand action space is another
//   `CommandParameterizing` type that needs the hand kinematics (open point A of
//   planner_loss.md).
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave:
// - Source/Python/lem/model/intention_generator.py lines 140-435 (the class and
//   generate_action entire): the facts above.
// - Source/Python/lem/tests/test_planner_selection.py lines 1-80: the exact scalar forward
//   model pattern ("value = action[:, 0, :1]", a goal of 0, strength_reg 0) and its pass
//   lines: the chosen command scores at or below the start and equals the minimum of the
//   seen scores; with one update a body at 1.0 moves to 0.9 and a body at 0.001 keeps it.
// - Source/Python/mimo_bridge/hand_target_control.py lines 392-439 (HandActionSpace).
// - apple/UEMAKit/Specs/planner_loss.md sections 1 to 3 and 6.
// - apple/UEMAKit/Sources/UEMACore/Protocols.swift: `Planning`, `PlanResult` (21
//   evaluations, 20 updates, learning rate 0.1, strength weight 0.01).
// - https://raw.githubusercontent.com/ml-explore/mlx-swift/main/Source/MLX/Transforms+Grad.swift
//   (fetched November 4, 2026): "public func valueAndGrad(_ f: @escaping ([MLXArray]) ->
//   [MLXArray], argumentNumbers: some Collection<Int> = [0]) -> ([MLXArray]) ->
//   ([MLXArray], [MLXArray])".
// - PyTorch torch/optim/adam.py, _single_tensor_adam (the update `TorchAdam` writes out):
//   "exp_avg.lerp_(grad, 1 - beta1)"; "exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 -
//   beta2)"; "step_size = lr / bias_correction1"; "denom = (exp_avg_sq.sqrt() /
//   bias_correction2_sqrt).add_(eps)"; "param.addcdiv_(exp_avg, denom, value=-step_size)";
//   defaults betas (0.9, 0.999), eps 1e-8 (recalled from the PyTorch source, not fetched in
//   this item).

import Foundation
import MLX
import UEMACore

// MARK: - Errors

/// The errors of a planning event.
public enum PlannerError: Error, Equatable, Sendable {
    /// The start command's score or numbers are not finite (intention_generator.py:373-375).
    case nonFiniteStart
    /// The goal ring holds no entry with a finite wellness.
    case noGoal
    /// The start commands and the goals differ in count, or there are none.
    case bodyCountMismatch(commands: Int, goals: Int)
    /// A command of the wrong scale was given (the planner takes `.normalized`).
    case wrongScale(found: CommandScale)
    /// A parameterization or objective returned arrays whose count or shape does not fit.
    case badParameters(String)
}

// MARK: - The forward model as the planner sees it

/// The differentiable map the search descends through: raw commands [B, 26, 9] (positions in
/// position_range units, before the execution transform) to the predicted next token of each
/// sense, [B, 192], built from MLX operations so that `valueAndGrad` reaches the command. The
/// concrete type holds this model step's tokens, the pre-shift context and the frozen
/// weights; it never updates them (specification REQ-B-17).
public protocol PlanTransition: AnyObject {
    /// The senses whose predicted token enters the goal term: `SenseName.goalScored` minus
    /// any sense whose input of this model step is not fresh (vision on a model step without
    /// a render, planner_loss.md section 3 step 5). Motor is never scored.
    var scoredSenses: [SenseName] { get }

    /// The predicted next tokens, one [B, 192] array per sense.
    func predictNext(rawCommands: MLXArray) -> [SenseName: MLXArray]
}

/// Gives the transition of one model step (the session's forward model holds the context
/// captured before this model step's shift, lem_model.py:6061, :6193).
public protocol PlanTransitionProviding: AnyObject {
    /// The transition for a planning event at `step`.
    func transition(at step: ModelStep) throws -> any PlanTransition
}

// MARK: - The search variables

/// Turns raw commands into the search's parameter arrays and back.
public protocol CommandParameterizing {
    /// The parameter arrays for `bodies` raw commands laid end to end (bodies x 234 numbers).
    func initialParameters(from rawCommands: [Float], bodies: Int) throws -> [MLXArray]

    /// The command [B, 26, 9] and the strengths [B, 26, 1] the parameters stand for, built
    /// from MLX operations (differentiable).
    func assemble(_ parameters: [MLXArray]) -> (command: MLXArray, strength: MLXArray)
}

/// The direct action space (`action_space is None`, intention_generator.py:244-252, 296-300):
/// four arrays, position [B, 26, 3], rotation [B, 26, 4], radius logit [B, 26, 1] and strength
/// logit [B, 26, 1].
public struct DirectCommandParameterization: CommandParameterizing {
    /// The clamp before a logit: 1e-4 (intention_generator.py:247-248).
    public static let logitClamp: Float = 1e-4

    /// The floor of a quaternion's norm before division: 1e-12 (torch.nn.functional.normalize's
    /// default eps).
    public static let normFloor: Float = 1e-12

    /// The parameterization has no settings.
    public init() {}

    /// log(v / (1 - v)) with v clamped to 1e-4 ... 1 - 1e-4.
    public static func logit(_ value: Float) -> Float {
        let v = min(max(value, logitClamp), 1 - logitClamp)
        return Float(Foundation.log(Double(v) / Double(1 - v)))
    }

    public func initialParameters(from rawCommands: [Float], bodies: Int) throws -> [MLXArray] {
        guard bodies >= 1, rawCommands.count == bodies * Command.count else {
            throw PlannerError.badParameters("expected \(bodies) x 234 numbers, found \(rawCommands.count)")
        }
        let handles = Command.handleCount
        let width = Command.numbersPerHandle
        var position: [Float] = []
        var rotation: [Float] = []
        var radius: [Float] = []
        var strength: [Float] = []
        position.reserveCapacity(bodies * handles * 3)
        rotation.reserveCapacity(bodies * handles * 4)
        radius.reserveCapacity(bodies * handles)
        strength.reserveCapacity(bodies * handles)
        for row in 0 ..< bodies * handles {
            let start = row * width
            position.append(contentsOf: rawCommands[start ..< start + 3])
            rotation.append(contentsOf: rawCommands[start + 3 ..< start + 7])
            radius.append(DirectCommandParameterization.logit(rawCommands[start + 7]))
            strength.append(DirectCommandParameterization.logit(rawCommands[start + 8]))
        }
        return [
            MLXArray(position, [bodies, handles, 3]),
            MLXArray(rotation, [bodies, handles, 4]),
            MLXArray(radius, [bodies, handles, 1]),
            MLXArray(strength, [bodies, handles, 1]),
        ]
    }

    public func assemble(_ parameters: [MLXArray]) -> (command: MLXArray, strength: MLXArray) {
        let position = parameters[0]
        let rotation = parameters[1]
        let norm = MLX.sqrt((rotation * rotation).sum(axis: -1, keepDims: true))
        let unitRotation = rotation / MLX.maximum(norm, MLXArray(DirectCommandParameterization.normFloor))
        let radius = MLX.sigmoid(parameters[2])
        let strength = MLX.sigmoid(parameters[3])
        return (concatenated([position, unitRotation, radius, strength], axis: -1), strength)
    }
}

// MARK: - Adam as torch.optim.Adam runs it

/// The update of torch.optim.Adam (amsgrad off, weight decay 0) over parameter groups held as
/// Swift arrays: m = m + (1 - 0.9)(g - m); v = 0.999 v + (1 - 0.999) g^2; p = p - (lr / (1 -
/// 0.9^t)) m / (sqrt(v) / sqrt(1 - 0.999^t) + 1e-8), with t the update count from 1.
public struct TorchAdam {
    /// The first-moment decay: 0.9 (torch default).
    public static let beta1: Double = 0.9
    /// The second-moment decay: 0.999 (torch default).
    public static let beta2: Double = 0.999
    /// The denominator floor: 1e-8 (torch default).
    public static let epsilon: Double = 1e-8

    /// The step size: 0.1 in both searches of this module.
    public let learningRate: Double

    /// Updates applied so far.
    public private(set) var stepCount = 0

    private var firstMoments: [[Float]]
    private var secondMoments: [[Float]]

    /// Zero moments for groups of the given element counts.
    public init(learningRate: Float, parameterCounts: [Int]) {
        self.learningRate = Double(learningRate)
        firstMoments = parameterCounts.map { Array(repeating: 0, count: $0) }
        secondMoments = parameterCounts.map { Array(repeating: 0, count: $0) }
    }

    /// One update of every group in place; `gradients` matches `parameters` group by group.
    public mutating func update(_ parameters: inout [[Float]], gradients: [[Float]]) {
        precondition(parameters.count == firstMoments.count && gradients.count == firstMoments.count,
                     "Adam: one gradient group per parameter group")
        stepCount += 1
        let t = Double(stepCount)
        let biasCorrection1 = 1 - Foundation.pow(TorchAdam.beta1, t)
        let biasCorrection2Root = (1 - Foundation.pow(TorchAdam.beta2, t)).squareRoot()
        let stepSize = Float(learningRate / biasCorrection1)
        let oneMinusBeta1 = Float(1 - TorchAdam.beta1)
        let beta2 = Float(TorchAdam.beta2)
        let oneMinusBeta2 = Float(1 - TorchAdam.beta2)
        let root2 = Float(biasCorrection2Root)
        let eps = Float(TorchAdam.epsilon)
        for group in parameters.indices {
            precondition(parameters[group].count == gradients[group].count
                         && parameters[group].count == firstMoments[group].count,
                         "Adam: group \(group) changed size")
            for i in parameters[group].indices {
                let g = gradients[group][i]
                let m = firstMoments[group][i] + oneMinusBeta1 * (g - firstMoments[group][i])
                let v = beta2 * secondMoments[group][i] + oneMinusBeta2 * g * g
                firstMoments[group][i] = m
                secondMoments[group][i] = v
                let denominator = v.squareRoot() / root2 + eps
                parameters[group][i] -= stepSize * m / denominator
            }
        }
    }
}

// MARK: - The search

/// The three score parts of one evaluation of one body (`parts[:, 0:3]`,
/// intention_generator.py:370-371).
public struct ScoreParts: Equatable, Sendable {
    /// goal + effort.
    public let total: Float
    /// The summed per-sense mean squared difference to the goal tokens.
    public let goal: Float
    /// 0.01 x the sum of squared strengths.
    public let effort: Float

    /// The parts as given.
    public init(total: Float, goal: Float, effort: Float) {
        self.total = total
        self.goal = goal
        self.effort = effort
    }

    /// True when all three are finite.
    public var isFinite: Bool { total.isFinite && goal.isFinite && effort.isFinite }
}

/// Why the search ended (the Python metadata's `stopped_by`).
public enum SearchStop: String, Equatable, Sendable {
    /// All updates ran (`step_budget`).
    case stepBudget = "step_budget"
    /// A score or command number was not finite (`nonfinite_candidate`).
    case nonfiniteCandidate = "nonfinite_candidate"
    /// A gradient number was not finite (`nonfinite_gradient`).
    case nonfiniteGradient = "nonfinite_gradient"
}

/// What one command search produced.
public struct CommandSearchResult: Equatable, Sendable {
    /// The bodies searched.
    public let bodies: Int
    /// The selected raw command of each body, 234 numbers (positions in position_range units).
    public let commands: [[Float]]
    /// The score parts of each evaluation (start first), one entry per body.
    public let evaluations: [[ScoreParts]]
    /// The index into `evaluations` of each body's selected evaluation.
    public let selectedIteration: [Int]
    /// Adam updates applied.
    public let updates: Int
    /// Why the search ended.
    public let stoppedBy: SearchStop

    /// The total of each evaluation for one body, start first (the plan panel's 21 scores).
    public func totals(body: Int) -> [Float] {
        evaluations.map { $0[body].total }
    }

    /// The total of the selected evaluation of one body.
    public func selectedTotal(body: Int) -> Float {
        evaluations[selectedIteration[body]][body].total
    }
}

/// One command search: from one `.normalized` start command per body toward one goal token
/// set per body through a transition, returning the selected command per body and every
/// evaluation's score parts. The planner depends on this protocol, not on the Adam type.
public protocol CommandSearching: AnyObject {
    /// Runs one search; mismatched counts, a wrong scale or a non-finite start throw.
    func run(start: [Command], goals: [TokenSet], transition: any PlanTransition) throws -> CommandSearchResult
}

/// Adam on the command through the frozen forward model (specification REQ-B-12; the plan's
/// 21 evaluations and 20 gradient passes).
public final class AdamCommandSearch: CommandSearching {
    /// Adam updates per search: 20 (`n_steps`, lem_model.py:1211; `PlanResult.updatesPerEvent`).
    public let updates: Int
    /// Adam's step size: 0.1 (`lr`, lem_model.py:1211; `PlanResult.learningRate`).
    public let learningRate: Float
    /// The weight of the sum of squared strengths: 0.01 (`strength_reg`, lem_model.py:1211).
    public let strengthRegularisation: Float

    private let parameterization: any CommandParameterizing

    /// A search over `parameterization`'s variables with the released settings by default.
    public init(
        parameterization: any CommandParameterizing = DirectCommandParameterization(),
        updates: Int = PlanResult.updatesPerEvent,
        learningRate: Float = PlanResult.learningRate,
        strengthRegularisation: Float = PlanResult.strengthRegularisation
    ) {
        precondition(updates >= 0, "updates is 0 or more; got \(updates)")
        self.parameterization = parameterization
        self.updates = updates
        self.learningRate = learningRate
        self.strengthRegularisation = strengthRegularisation
    }

    /// Runs one search from `start` (one `.normalized` command per body) toward `goals` (one
    /// token set per body) through `transition`.
    public func run(start: [Command], goals: [TokenSet], transition: any PlanTransition) throws -> CommandSearchResult {
        guard !start.isEmpty, start.count == goals.count else {
            throw PlannerError.bodyCountMismatch(commands: start.count, goals: goals.count)
        }
        if let wrong = start.first(where: { $0.scale != .normalized }) {
            throw PlannerError.wrongScale(found: wrong.scale)
        }
        let bodies = start.count
        let width = Command.count
        var parameters = try parameterization.initialParameters(from: start.flatMap { $0.values }, bodies: bodies)
        let shapes = parameters.map { $0.shape }

        // A sense enters the goal term when the transition scores it and every body's goal
        // entry holds its token (intention_generator.py:343-346: "m in desired_encoded").
        var goalArrays: [SenseName: MLXArray] = [:]
        var senses: [SenseName] = []
        for sense in transition.scoredSenses where sense != .motor {
            var flat: [Float] = []
            flat.reserveCapacity(bodies * Token.count)
            var complete = true
            for goal in goals {
                guard let token = goal[sense] else {
                    complete = false
                    break
                }
                flat.append(contentsOf: token.values)
            }
            if complete {
                senses.append(sense)
                goalArrays[sense] = MLXArray(flat, [bodies, Token.count])
            }
        }
        let goalTokens = goalArrays
        let scored = senses
        let parameterization = self.parameterization
        let weight = MLXArray(strengthRegularisation)
        let zeros = MLXArray(Array(repeating: Float(0), count: bodies), [bodies])

        // Outputs: [mean total (the differentiated loss), total [B], goal [B], effort [B],
        // command [B, 26, 9]].
        let objective: ([MLXArray]) -> [MLXArray] = { values in
            let assembled = parameterization.assemble(values)
            let predicted = transition.predictNext(rawCommands: assembled.command)
            var goal = zeros
            for sense in scored {
                guard let prediction = predicted[sense], let target = goalTokens[sense] else { continue }
                let difference = prediction.reshaped([bodies, -1]) - target
                goal = goal + (difference * difference).mean(axis: -1)
            }
            let strength = assembled.strength.reshaped([bodies, -1])
            let effort = (strength * strength).sum(axis: -1) * weight
            let total = goal + effort
            return [total.mean(), total, goal, effort, assembled.command]
        }
        let lossAndGradient = valueAndGrad(objective, argumentNumbers: Array(0 ..< parameters.count))

        var adam = TorchAdam(learningRate: learningRate, parameterCounts: shapes.map { $0.reduce(1, *) })
        var evaluations: [[ScoreParts]] = []
        var best: [ScoreParts] = []
        var bestCommands: [[Float]] = []
        var bestStep = Array(repeating: 0, count: bodies)
        var applied = 0
        var stop = SearchStop.stepBudget

        for iteration in 0 ... updates {
            let wantsGradient = iteration < updates
            let outputs: [MLXArray]
            var gradients: [MLXArray] = []
            if wantsGradient {
                let (values, grads) = lossAndGradient(parameters)
                outputs = values
                gradients = grads
            } else {
                outputs = objective(parameters)
            }
            guard outputs.count == 5 else {
                throw PlannerError.badParameters("the objective gave \(outputs.count) outputs, 5 expected")
            }
            let totals = outputs[1].asArray(Float.self)
            let goalValues = outputs[2].asArray(Float.self)
            let efforts = outputs[3].asArray(Float.self)
            let commandValues = outputs[4].asArray(Float.self)
            guard totals.count == bodies, goalValues.count == bodies, efforts.count == bodies,
                  commandValues.count == bodies * width else {
                throw PlannerError.badParameters(
                    "the objective gave \(totals.count) totals and \(commandValues.count) command numbers for \(bodies) bodies")
            }
            let parts = (0 ..< bodies).map {
                ScoreParts(total: totals[$0], goal: goalValues[$0], effort: efforts[$0])
            }
            let finite = (0 ..< bodies).map { body in
                parts[body].isFinite
                    && commandValues[body * width ..< (body + 1) * width].allSatisfy { $0.isFinite }
            }
            if iteration == 0 {
                guard finite.allSatisfy({ $0 }) else { throw PlannerError.nonFiniteStart }
                best = parts
                bestCommands = (0 ..< bodies).map { Array(commandValues[$0 * width ..< ($0 + 1) * width]) }
            } else {
                for body in 0 ..< bodies where finite[body] && parts[body].total < best[body].total {
                    best[body] = parts[body]
                    bestCommands[body] = Array(commandValues[body * width ..< (body + 1) * width])
                    bestStep[body] = iteration
                }
            }
            evaluations.append(parts)
            if !finite.allSatisfy({ $0 }) {
                stop = .nonfiniteCandidate
                break
            }
            guard wantsGradient else { break }
            let gradientValues = gradients.map { $0.asArray(Float.self) }
            guard gradientValues.count == parameters.count else {
                throw PlannerError.badParameters("\(gradientValues.count) gradients for \(parameters.count) parameter arrays")
            }
            if gradientValues.contains(where: { group in group.contains { !$0.isFinite } }) {
                stop = .nonfiniteGradient
                break
            }
            var values = parameters.map { $0.asArray(Float.self) }
            adam.update(&values, gradients: gradientValues)
            parameters = zip(values, shapes).map { MLXArray($0.0, $0.1) }
            applied += 1
        }

        return CommandSearchResult(
            bodies: bodies, commands: bestCommands, evaluations: evaluations,
            selectedIteration: bestStep, updates: applied, stoppedBy: stop)
    }
}

// MARK: - The planner

/// What the last planning event used and produced, for the plan panel (specification
/// REQ-B-14: goal ring fill, the goal's wellness and rank, the 21 scores with the selected
/// iteration, the hand descent's scores).
public struct PlannerEventRecord: Equatable, Sendable {
    /// The model step of the event.
    public let step: ModelStep
    /// The goal entry's index among the ring's entries (oldest first).
    public let goalIndex: Int
    /// The goal entry's rank by wellness, 1 for the highest.
    public let goalRank: Int?
    /// The ring's entry count at the event.
    public let goalRingCount: Int
    /// The command search.
    public let search: CommandSearchResult
    /// The hand descent, when it ran.
    public let handDescent: HandDescentResult?
}

/// The planner (plan row WI-39): records each model step into the goal ring and, on a
/// planning model step, runs the command search toward the goal entry and then the hand
/// descent when one is injected. Implements `Planning` for one body.
public final class Planner: Planning {
    private let ring: GoalRing
    private let goalRule: any GoalSelecting
    private let transitions: any PlanTransitionProviding
    private let search: any CommandSearching
    private let handDescent: (any HandDescending)?

    /// The last planning event; nil before the first or after `reset()`.
    public private(set) var lastEvent: PlannerEventRecord?

    /// A planner over an injected ring, goal rule, transition source, search and hand descent
    /// (nil runs no hand descent).
    public init(
        ring: GoalRing,
        goalRule: any GoalSelecting,
        transitions: any PlanTransitionProviding,
        search: any CommandSearching,
        handDescent: (any HandDescending)?
    ) {
        self.ring = ring
        self.goalRule = goalRule
        self.transitions = transitions
        self.search = search
        self.handDescent = handDescent
    }

    public var goalRingCount: Int { ring.count }

    /// The goal ring's entries, oldest first.
    public var goalRingEntries: [GoalRingEntry] { ring.entries }

    public func record(tokens: TokenSet, wellness: Float, at step: ModelStep) {
        ring.append(tokens: tokens, wellness: wellness, at: step)
    }

    public func plan(from command: Command, at step: ModelStep) throws -> PlanResult {
        guard command.scale == .normalized else {
            throw PlannerError.wrongScale(found: command.scale)
        }
        guard let goal = ring.goal(by: goalRule) else {
            throw PlannerError.noGoal
        }
        let transition = try transitions.transition(at: step)
        let result = try search.run(start: [command], goals: [goal.entry.tokens], transition: transition)
        guard result.bodies == 1, let selectedValues = result.commands.first,
              let selected = result.selectedIteration.first else {
            throw PlannerError.badParameters("the search returned \(result.commands.count) commands for 1 body")
        }
        let chosen = try Command(values: selectedValues, scale: .normalized)
        let hand = try handDescent?.descend(after: chosen, goal: goal.entry, at: step)
        lastEvent = PlannerEventRecord(
            step: step, goalIndex: goal.index, goalRank: ring.wellnessRank(ofEntryAt: goal.index),
            goalRingCount: ring.count, search: result, handDescent: hand)
        return try PlanResult(
            command: chosen,
            scores: result.totals(body: 0),
            selectedIteration: selected,
            goalStep: goal.entry.step,
            goalWellness: goal.entry.wellness,
            fingerGoals: hand?.fingerGoals)
    }

    public func reset() {
        ring.removeAll()
        lastEvent = nil
    }
}
