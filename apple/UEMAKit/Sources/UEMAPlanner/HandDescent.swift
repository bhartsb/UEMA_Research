// HandDescent.swift: the hand descent. After the command search, Adam on the 44 finger joint
// goals (radians, the hand hinge joints) for 20 iterations, scored by the calibrated linear
// hand response over a 250 ms horizon, inside an 11 degree bound around the starting goal.
// Work item WI-39, target B of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html
// (specification REQ-B-12; the specification's section 5 names the concrete type
// JointHandDescent).
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). The "First build on a Mac" checklist of this work item
// is in Planner.swift's header.
//
// INPUT, OPERATION, OUTPUT (apple/UEMAKit/Specs/planner_loss.md section 4;
// Source/Python/lem/model/hand_response.py:128-199, `choose_hand_goal`):
// - Input per body: the starting raw joint goal r [44] (the hand action space's goal joints),
//   the current hand angles a [44] (radians) and velocities w [44] (rad/s), the previous
//   joint goal p [44] (nil on the first event), the smoothing 0.35, the joint limits
//   [44 x 2], a validity flag (a remembered pose exists for the goal entry), and a cost
//   function of the executed goal.
// - Bounded goal of the parameters x: g = r + clip(x - r, -11 deg, +11 deg), then clamped
//   to the joint limits, then scaled into the hand's coupled-joint constraints along the
//   segment from r (physical_goal.py, PhysicalTrustRegion.goal_joints; the scaling is the
//   injected `HandJointProjecting.scaleIntoConstraints`). Executed goal: e = P(P(g)), and with
//   a previous goal e = P(0.35 P(g) + 0.65 p), P the hand kinematics' joint projection
//   (hand_response.py:162-165: `goal=kin.project_joints(bounded.goal_joints())`, the blend
//   when a previous goal exists and smoothing > 0, then `return kin.project_joints(goal)`).
// - Cost per body: c = cost(e) where the body is valid, 0 where it is not (`torch.where(valid,
//   costs, zeros_like(costs))`, hand_response.py:166-171; here MLX's `which`).
// - Loop: evaluation 0 at x = r; then 20 times: the gradient of the sum of c over bodies with
//   respect to x, one Adam update at learning rate 0.1 (hand_response.py:172, 181-183), the
//   evaluation of the updated x; a body's selection changes only when its cost is strictly
//   below its best and the body is valid (:185-189); a non-finite cost after an update throws
//   (:184).
// - Output per body: the selected executed goal e (already smoothed: the decoder takes it as
//   is), the selected bounded raw goal g, the 21 costs and the selected evaluation.
//
// THE COST. `CalibratedHandGoalCost` is `local_goal_cost` (hand_response.py:118-125): the
// linear response predicts the 250 ms joint angles q = P(a + range x (k0 + k1 (e - a) / range
// + k2 x 0.25 w / range)) with the receipt's coefficients k [44 x 3] (features intercept,
// goal minus initial over range, quarter-second velocity over range; hand_response.py:96-104),
// and the injected `HandPoseErrorMeasuring` turns q into the per-body pose error of the
// finger handles against the remembered target pose (`pose_error_per_body`,
// physical_goal.py:20-25: the squared position error over 1.80 cm plus the squared rotation
// error over 11.0 degrees, averaged over handles and axes). That error needs the hand
// kinematics' world assembly, which is not ported here (planner_loss.md open point B).
//
// DIFFERENCES FROM THE PYTHON, each named:
// - The Python runs the response in float64 (hand_response.py:97-100); MLX on the Apple GPU
//   computes in float32. The size of the difference on the 21 costs is unmeasured; the WI-34
//   fixture comparison measures it.
// - The Python scores each iteration twice (once for the gradient, once after the update,
//   hand_response.py:182-184); this file scores each parameter value once and takes the
//   gradient from the same call. The costs and updates are the same numbers.
// - The Python's check that the baseline rebuilt from the starting goal equals the command's
//   finger rows within 2e-5 (hand_response.py:174-180) needs the hand kinematics and is left
//   to the problem source.
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave:
// - Source/Python/lem/model/hand_response.py lines 1-220 (predict, local_goal_cost,
//   choose_hand_goal entire): the facts above; receipt check "choice ... steps=20, lr=.1,
//   raw_joint_bound_deg=11." (:51-52) and "horizon_ms ... 250" (:49-50).
// - Source/Python/lem/model/physical_goal.py (entire, 180 lines): pose_error_per_body and
//   PhysicalTrustRegion.goal_joints.
// - Source/Python/lem/model/efference.py lines 101-102: "pos_tenure_cm": 1.80,
//   "rot_tenure_deg": 11.0.
// - Source/Python/mimo_bridge/hand_target_control.py lines 392-439: HandActionSpace keeps
//   `initial` (the starting raw goal r) and descends on `joints` started at r.
// - apple/UEMAKit/Sources/UEMACore/Command.swift: `FingerJointGoals` (44 angles, radians).
// - https://raw.githubusercontent.com/ml-explore/mlx-swift/main/Source/MLX/Ops.swift (fetched
//   November 4, 2026): "public func which(_ condition: MLXArray, _ a: some ScalarOrArray, _ b:
//   some ScalarOrArray, stream: StreamOrDevice = .default) -> MLXArray", documented as an
//   alias of `where`; `maximum` and `minimum` take "_ a: some ScalarOrArray, _ b: some
//   ScalarOrArray".

import Foundation
import MLX
import UEMACore

// MARK: - Errors

/// The errors of a hand descent.
public enum HandDescentError: Error, Equatable, Sendable {
    /// An array of the problem has the wrong count (`name`, expected, found).
    case wrongCount(name: String, expected: Int, found: Int)
    /// An input number is not finite (hand_response.py:149-150).
    case nonFiniteInput(String)
    /// A cost after an update is not finite (hand_response.py:184).
    case nonFiniteSearch(iteration: Int)
    /// The smoothing lies outside 0 ... 1 (hand_response.py:148).
    case invalidSmoothing(Float)
    /// A problem with fewer than 1 body.
    case noBodies
    /// A body index outside the outcome's bodies (`body`, body count).
    case bodyOutOfRange(body: Int, bodies: Int)
}

// MARK: - The hand kinematics as the descent sees it

/// The hand kinematics' joint operations, built from MLX operations (differentiable).
public protocol HandJointProjecting {
    /// `HandKinematics.project_joints`: joint angles [B, 44] to the nearest feasible angles
    /// [B, 44] (hand_target_control.py:184).
    func projectJoints(_ joints: MLXArray) -> MLXArray

    /// The coupled-joint constraint scaling of PhysicalTrustRegion.goal_joints
    /// (physical_goal.py, the `for group in kin.groups` loop): each group's change from
    /// `initial` [B, 44] to `proposed` [B, 44] scaled by the largest factor in 0 ... 1 that
    /// keeps every constraint of the group within its limits.
    func scaleIntoConstraints(initial: MLXArray, proposed: MLXArray) -> MLXArray
}

/// The cost of executed joint goals.
public protocol HandGoalCosting {
    /// The cost per body, [B], of executed joint goals [B, 44] (radians), built from MLX
    /// operations (differentiable).
    func cost(executedGoal: MLXArray) -> MLXArray
}

/// The pose error of predicted joint angles against the remembered target pose.
public protocol HandPoseErrorMeasuring {
    /// The error per body, [B], of predicted joint angles [B, 44] (the finger handles' world
    /// poses assembled from them, compared with the target by `pose_error_per_body`).
    func error(predictedJoints: MLXArray) -> MLXArray
}

// MARK: - The calibrated linear hand response

/// The calibrated linear hand response (hand_response.py:96-104): the joint angles 250 ms
/// after a goal is given.
public struct LinearHandResponse: Equatable, Sendable {
    /// The hand hinge joints: 44.
    public static let jointCount = FingerJointGoals.count
    /// The features per joint: 3 (intercept, goal minus initial over range, quarter-second
    /// velocity over range).
    public static let featureCount = 3
    /// The response horizon: 250 ms (hand_response.py:49-50).
    public static let horizonMilliseconds = 250

    /// The receipt's coefficients, 44 x 3 row-major (joint, then feature).
    public let coefficients: [Float]
    /// Each joint's range, upper limit minus lower limit, radians (hand_response.py:69).
    public let ranges: [Float]

    /// The coefficient columns split once at init: intercept, goal minus initial over range,
    /// quarter-second velocity over range (44 each).
    private let intercept: [Float]
    private let goalWeight: [Float]
    private let velocityWeight: [Float]

    /// A response from the receipt's coefficients and the joint ranges; wrong counts or
    /// non-finite numbers throw.
    public init(coefficients: [Float], ranges: [Float]) throws {
        let joints = LinearHandResponse.jointCount
        guard coefficients.count == joints * LinearHandResponse.featureCount else {
            throw HandDescentError.wrongCount(
                name: "coefficients", expected: joints * LinearHandResponse.featureCount, found: coefficients.count)
        }
        guard ranges.count == joints else {
            throw HandDescentError.wrongCount(name: "ranges", expected: joints, found: ranges.count)
        }
        guard coefficients.allSatisfy({ $0.isFinite }), ranges.allSatisfy({ $0.isFinite && $0 != 0 }) else {
            throw HandDescentError.nonFiniteInput("hand response coefficients or ranges")
        }
        self.coefficients = coefficients
        self.ranges = ranges
        intercept = (0 ..< joints).map { coefficients[$0 * LinearHandResponse.featureCount] }
        goalWeight = (0 ..< joints).map { coefficients[$0 * LinearHandResponse.featureCount + 1] }
        velocityWeight = (0 ..< joints).map { coefficients[$0 * LinearHandResponse.featureCount + 2] }
    }

    /// The predicted angles [B, 44]: P(a + range x (k0 + k1 (goal - a) / range + k2 x 0.25 w /
    /// range)), from current angles a, velocities w and goals, each [B, 44].
    public func predictedJoints(
        initial: MLXArray, velocity: MLXArray, goal: MLXArray, projection: any HandJointProjecting
    ) -> MLXArray {
        let joints = LinearHandResponse.jointCount
        let range = MLXArray(ranges, [joints])
        let change = MLXArray(intercept, [joints])
            + MLXArray(goalWeight, [joints]) * ((goal - initial) / range)
            + MLXArray(velocityWeight, [joints]) * (MLXArray(Float(0.25)) * velocity / range)
        return projection.projectJoints(initial + change * range)
    }
}

/// `local_goal_cost` (hand_response.py:118-125) for one event: the response's predicted
/// angles from this event's current angles and velocities, measured by the pose error.
public struct CalibratedHandGoalCost: HandGoalCosting {
    private let response: LinearHandResponse
    private let initial: MLXArray
    private let velocity: MLXArray
    private let projection: any HandJointProjecting
    private let poseError: any HandPoseErrorMeasuring

    /// The cost from `bodies` x 44 current angles (radians) and velocities (rad/s).
    public init(
        response: LinearHandResponse,
        currentAngles: [Float],
        currentVelocities: [Float],
        bodies: Int,
        projection: any HandJointProjecting,
        poseError: any HandPoseErrorMeasuring
    ) throws {
        let count = bodies * LinearHandResponse.jointCount
        guard currentAngles.count == count else {
            throw HandDescentError.wrongCount(name: "currentAngles", expected: count, found: currentAngles.count)
        }
        guard currentVelocities.count == count else {
            throw HandDescentError.wrongCount(name: "currentVelocities", expected: count, found: currentVelocities.count)
        }
        guard currentAngles.allSatisfy({ $0.isFinite }), currentVelocities.allSatisfy({ $0.isFinite }) else {
            throw HandDescentError.nonFiniteInput("current hand angles or velocities")
        }
        self.response = response
        initial = MLXArray(currentAngles, [bodies, LinearHandResponse.jointCount])
        velocity = MLXArray(currentVelocities, [bodies, LinearHandResponse.jointCount])
        self.projection = projection
        self.poseError = poseError
    }

    public func cost(executedGoal: MLXArray) -> MLXArray {
        poseError.error(predictedJoints: response.predictedJoints(
            initial: initial, velocity: velocity, goal: executedGoal, projection: projection))
    }
}

// MARK: - One problem and its result

/// Everything one hand descent needs, for `bodies` bodies (arrays laid end to end, 44 per body).
public struct HandDescentProblem {
    /// The bodies.
    public let bodies: Int
    /// The starting raw joint goal r, radians.
    public let rawInitialGoal: [Float]
    /// The previous executed joint goal, radians; nil on the first event.
    public let previousGoal: [Float]?
    /// The weight of the new goal in the blend with the previous one: 0.35 (`smoothing`,
    /// `smooth_actions` 0.35, planner_loss.md section 3 step 2).
    public let smoothing: Float
    /// The lower joint limits, 44 radians.
    public let lowerLimits: [Float]
    /// The upper joint limits, 44 radians.
    public let upperLimits: [Float]
    /// One flag per body: true when the goal entry has a remembered pose.
    public let valid: [Bool]
    /// The cost of executed goals.
    public let cost: any HandGoalCosting
    /// The hand kinematics' joint operations.
    public let projection: any HandJointProjecting

    /// A problem; counts are checked by `JointHandDescent.run(_:)`.
    public init(
        bodies: Int, rawInitialGoal: [Float], previousGoal: [Float]?, smoothing: Float = 0.35,
        lowerLimits: [Float], upperLimits: [Float], valid: [Bool],
        cost: any HandGoalCosting, projection: any HandJointProjecting
    ) {
        self.bodies = bodies
        self.rawInitialGoal = rawInitialGoal
        self.previousGoal = previousGoal
        self.smoothing = smoothing
        self.lowerLimits = lowerLimits
        self.upperLimits = upperLimits
        self.valid = valid
        self.cost = cost
        self.projection = projection
    }
}

/// What one hand descent produced for one body (the planner's single body): the selected
/// executed goal for the decoder, the selected bounded raw goal, the 21 costs and the
/// selected evaluation (the plan panel's "hand descent's 20 scores", specification REQ-B-14,
/// with the start's cost first).
public struct HandDescentResult: Equatable, Sendable {
    /// The selected executed joint goal (already smoothed).
    public let fingerGoals: FingerJointGoals
    /// The selected bounded raw joint goal, radians.
    public let rawGoal: [Double]
    /// The cost of each evaluation, start first.
    public let costs: [Float]
    /// The index into `costs` of the selected evaluation.
    public let selectedIteration: Int

    /// A result as given.
    public init(fingerGoals: FingerJointGoals, rawGoal: [Double], costs: [Float], selectedIteration: Int) {
        self.fingerGoals = fingerGoals
        self.rawGoal = rawGoal
        self.costs = costs
        self.selectedIteration = selectedIteration
    }
}

/// The per-body outcome of `JointHandDescent.run(_:)`.
public struct HandDescentOutcome: Equatable, Sendable {
    /// The selected executed goal per body, 44 radians each.
    public let executedGoals: [[Float]]
    /// The selected bounded raw goal per body, 44 radians each.
    public let rawGoals: [[Float]]
    /// The costs of each evaluation (start first), one per body.
    public let costs: [[Float]]
    /// The selected evaluation per body.
    public let selectedIteration: [Int]

    /// The result of one body; a body outside the outcome throws.
    public func result(body: Int) throws -> HandDescentResult {
        guard executedGoals.indices.contains(body), rawGoals.indices.contains(body),
              selectedIteration.indices.contains(body), costs.allSatisfy({ $0.indices.contains(body) }) else {
            throw HandDescentError.bodyOutOfRange(body: body, bodies: executedGoals.count)
        }
        return HandDescentResult(
            fingerGoals: try FingerJointGoals(angles: executedGoals[body].map { Double($0) }),
            rawGoal: rawGoals[body].map { Double($0) },
            costs: costs.map { $0[body] },
            selectedIteration: selectedIteration[body])
    }
}

// MARK: - The planner's view of the hand descent

/// Runs the hand descent after the command search of a planning event; nil when it does
/// not run (no remembered pose for the goal entry).
public protocol HandDescending: AnyObject {
    /// The hand descent after the selected command `command` toward `goal`.
    func descend(after command: Command, goal: GoalRingEntry, at step: ModelStep) throws -> HandDescentResult?
}

/// Builds the problem of one event from the session's state (current hand angles and
/// velocities, the previous goal, the remembered pose of the goal entry, the hand
/// kinematics); nil when the goal entry has no remembered pose.
public protocol HandDescentProblemSource: AnyObject {
    /// The problem after `command` toward `goal` at `step`, for one body.
    func problem(after command: Command, goal: GoalRingEntry, at step: ModelStep) throws -> HandDescentProblem?
}

// MARK: - The descent

/// Adam on the 44 finger joint goals (hand_response.py:128-199).
public final class JointHandDescent: HandDescending {
    /// Adam updates: 20 (receipt `choice.steps`, hand_response.py:51).
    public let updates: Int
    /// Adam's step size: 0.1 (receipt `choice.lr`, hand_response.py:51, :172).
    public let learningRate: Float
    /// The bound on each joint's change from the starting goal: 11.0 degrees (receipt
    /// `choice.raw_joint_bound_deg`; `rot_tenure_deg`, efference.py:102).
    public let boundDegrees: Float

    private let problems: any HandDescentProblemSource

    /// A descent over the problems `problems` builds, with the released settings by default.
    public init(
        problems: any HandDescentProblemSource,
        updates: Int = 20,
        learningRate: Float = 0.1,
        boundDegrees: Float = 11.0
    ) {
        precondition(updates >= 0, "updates is 0 or more; got \(updates)")
        self.problems = problems
        self.updates = updates
        self.learningRate = learningRate
        self.boundDegrees = boundDegrees
    }

    public func descend(after command: Command, goal: GoalRingEntry, at step: ModelStep) throws -> HandDescentResult? {
        guard let problem = try problems.problem(after: command, goal: goal, at: step) else { return nil }
        return try run(problem).result(body: 0)
    }

    /// Runs one descent.
    public func run(_ problem: HandDescentProblem) throws -> HandDescentOutcome {
        let joints = LinearHandResponse.jointCount
        let bodies = problem.bodies
        guard bodies >= 1 else { throw HandDescentError.noBodies }
        let count = bodies * joints
        try check(problem.rawInitialGoal, name: "rawInitialGoal", count: count)
        if let previous = problem.previousGoal {
            try check(previous, name: "previousGoal", count: count)
        }
        try check(problem.lowerLimits, name: "lowerLimits", count: joints)
        try check(problem.upperLimits, name: "upperLimits", count: joints)
        guard problem.valid.count == bodies else {
            throw HandDescentError.wrongCount(name: "valid", expected: bodies, found: problem.valid.count)
        }
        guard problem.smoothing >= 0, problem.smoothing <= 1 else {
            throw HandDescentError.invalidSmoothing(problem.smoothing)
        }

        let shape = [bodies, joints]
        let raw = MLXArray(problem.rawInitialGoal, shape)
        let previous = problem.previousGoal.map { MLXArray($0, shape) }
        let smoothing = problem.smoothing
        let lower = MLXArray(problem.lowerLimits, [joints])
        let upper = MLXArray(problem.upperLimits, [joints])
        let bound = boundDegrees * Float.pi / 180
        let boundHigh = MLXArray(bound)
        let boundLow = MLXArray(-bound)
        let validMask = MLXArray(problem.valid, [bodies])
        let cost = problem.cost
        let projection = problem.projection

        // Outputs: [sum of costs (the differentiated loss), costs [B], executed [B, 44],
        // bounded raw [B, 44]].
        let objective: ([MLXArray]) -> [MLXArray] = { values in
            let delta = MLX.minimum(MLX.maximum(values[0] - raw, boundLow), boundHigh)
            let limited = MLX.maximum(MLX.minimum(raw + delta, upper), lower)
            let bounded = projection.scaleIntoConstraints(initial: raw, proposed: limited)
            // hand_response.py:162-165: project, blend with the previous goal, project again.
            var goal = projection.projectJoints(bounded)
            if let previous, smoothing > 0 {
                goal = MLXArray(smoothing) * goal + MLXArray(1 - smoothing) * previous
            }
            let executed = projection.projectJoints(goal)
            // hand_response.py:171: an invalid body's cost is 0, also when its cost is NaN.
            let costs = which(validMask, cost.cost(executedGoal: executed), Float(0))
            return [costs.sum(), costs, executed, bounded]
        }
        let lossAndGradient = valueAndGrad(objective, argumentNumbers: [0])

        var parameters = problem.rawInitialGoal
        var adam = TorchAdam(learningRate: learningRate, parameterCounts: [count])
        var costTrace: [[Float]] = []
        var best: [Float] = []
        var bestExecuted: [[Float]] = []
        var bestRaw: [[Float]] = []
        var selected = Array(repeating: 0, count: bodies)

        for iteration in 0 ... updates {
            let wantsGradient = iteration < updates
            let current = MLXArray(parameters, shape)
            let outputs: [MLXArray]
            var gradient: [Float] = []
            if wantsGradient {
                let (values, grads) = lossAndGradient([current])
                outputs = values
                gradient = grads.first?.asArray(Float.self) ?? []
            } else {
                outputs = objective([current])
            }
            let costs = outputs[1].asArray(Float.self)
            let executed = outputs[2].asArray(Float.self)
            let bounded = outputs[3].asArray(Float.self)
            guard costs.count == bodies, executed.count == count, bounded.count == count else {
                throw HandDescentError.wrongCount(name: "objective outputs", expected: count, found: executed.count)
            }
            if iteration == 0 {
                best = costs
                bestExecuted = (0 ..< bodies).map { Array(executed[$0 * joints ..< ($0 + 1) * joints]) }
                bestRaw = (0 ..< bodies).map { Array(bounded[$0 * joints ..< ($0 + 1) * joints]) }
            } else {
                guard costs.allSatisfy({ $0.isFinite }) else {
                    throw HandDescentError.nonFiniteSearch(iteration: iteration)
                }
                for body in 0 ..< bodies where problem.valid[body] && costs[body] < best[body] {
                    best[body] = costs[body]
                    bestExecuted[body] = Array(executed[body * joints ..< (body + 1) * joints])
                    bestRaw[body] = Array(bounded[body * joints ..< (body + 1) * joints])
                    selected[body] = iteration
                }
            }
            costTrace.append(costs)
            guard wantsGradient else { break }
            guard gradient.count == count else {
                throw HandDescentError.wrongCount(name: "gradient", expected: count, found: gradient.count)
            }
            var groups = [parameters]
            adam.update(&groups, gradients: [gradient])
            parameters = groups[0]
        }

        return HandDescentOutcome(
            executedGoals: bestExecuted, rawGoals: bestRaw, costs: costTrace, selectedIteration: selected)
    }

    private func check(_ values: [Float], name: String, count: Int) throws {
        guard values.count == count else {
            throw HandDescentError.wrongCount(name: name, expected: count, found: values.count)
        }
        guard values.allSatisfy({ $0.isFinite }) else {
            throw HandDescentError.nonFiniteInput(name)
        }
    }
}
