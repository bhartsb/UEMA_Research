// HandDescentTests.swift: the hand descent and the linear hand response (work item WI-39).
// Nothing in this file was compiled on the machine that wrote it (no Swift toolchain on
// Windows). Pass lines from Source/Python/lem/model/hand_response.py:128-199 (20 Adam
// updates at learning rate 0.1, best cost kept, invalid bodies unchanged, 0.35 blend with
// the previous goal) and physical_goal.py (the 11 degree bound around the starting goal).

import XCTest
import MLX
import UEMACore
@testable import UEMAPlanner

/// Joint operations that change nothing: no projection, no coupled constraints.
private struct IdentityProjection: HandJointProjecting {
    func projectJoints(_ joints: MLXArray) -> MLXArray { joints }
    func scaleIntoConstraints(initial: MLXArray, proposed: MLXArray) -> MLXArray { proposed }
}

/// A projection that halves every angle, so that the number of projections applied shows in
/// the executed goal (hand_response.py:162-165 projects before and after the blend).
private struct HalvingProjection: HandJointProjecting {
    func projectJoints(_ joints: MLXArray) -> MLXArray { joints * MLXArray(Float(0.5)) }
    func scaleIntoConstraints(initial: MLXArray, proposed: MLXArray) -> MLXArray { proposed }
}

/// The squared distance per body plus a fixed per-body addend (NaN for a body marks it as
/// unscorable).
private struct DistancePlusAddendCost: HandGoalCosting {
    let target: MLXArray
    let addend: MLXArray
    func cost(executedGoal: MLXArray) -> MLXArray {
        let difference = executedGoal - target
        return (difference * difference).sum(axis: -1) + addend
    }
}

/// The squared distance per body to a fixed target goal: sum over 44 joints of (e - t)^2.
private struct DistanceCost: HandGoalCosting {
    let target: MLXArray
    func cost(executedGoal: MLXArray) -> MLXArray {
        let difference = executedGoal - target
        return (difference * difference).sum(axis: -1)
    }
}

/// The squared distance per body of predicted joints to a fixed target.
private struct DistanceError: HandPoseErrorMeasuring {
    let target: MLXArray
    func error(predictedJoints: MLXArray) -> MLXArray {
        let difference = predictedJoints - target
        return (difference * difference).sum(axis: -1)
    }
}

final class HandDescentTests: XCTestCase {
    private let joints = FingerJointGoals.count

    private func problem(
        bodies: Int = 1, start: Float = 0, target: Float, previous: [Float]? = nil, valid: [Bool]? = nil
    ) -> HandDescentProblem {
        HandDescentProblem(
            bodies: bodies,
            rawInitialGoal: Array(repeating: start, count: bodies * joints),
            previousGoal: previous,
            smoothing: 0.35,
            lowerLimits: Array(repeating: -3, count: joints),
            upperLimits: Array(repeating: 3, count: joints),
            valid: valid ?? Array(repeating: true, count: bodies),
            cost: DistanceCost(target: MLXArray(Array(repeating: target, count: bodies * joints), [bodies, joints])),
            projection: IdentityProjection())
    }

    /// 21 costs, the selected one the lowest and at or below the start.
    func testTwentyOneCostsAndTheSelectedIsTheLowest() throws {
        let outcome = try JointHandDescent(problems: NoProblems()).run(problem(target: 0.05))
        XCTAssertEqual(outcome.costs.count, 21)
        let costs = outcome.costs.map { $0[0] }
        let selected = costs[outcome.selectedIteration[0]]
        XCTAssertLessThanOrEqual(selected, costs[0])
        XCTAssertEqual(selected, try XCTUnwrap(costs.min()), accuracy: 0)
    }

    /// A target 1 rad away: the raw goal moves at most 11 degrees (0.19199 rad) per joint.
    func testRawGoalStaysWithinElevenDegrees() throws {
        let outcome = try JointHandDescent(problems: NoProblems()).run(problem(target: 1))
        let bound = Float(11) * Float.pi / 180
        for value in outcome.rawGoals[0] {
            XCTAssertLessThanOrEqual(abs(value), bound + 1e-6)
        }
        XCTAssertEqual(try XCTUnwrap(outcome.rawGoals[0].max()), bound, accuracy: 1e-5)
    }

    /// An invalid body keeps its start: cost 0 at every evaluation, selected evaluation 0.
    func testInvalidBodyKeepsItsStart() throws {
        let outcome = try JointHandDescent(problems: NoProblems())
            .run(problem(bodies: 2, target: 0.05, valid: [false, true]))
        XCTAssertEqual(outcome.selectedIteration[0], 0)
        XCTAssertTrue(outcome.costs.allSatisfy { $0[0] == 0 })
        XCTAssertEqual(outcome.executedGoals[0], Array(repeating: 0, count: joints))
        XCTAssertGreaterThan(outcome.selectedIteration[1], 0)
    }

    /// With a previous goal p the executed goal is 0.35 g + 0.65 p (hand_response.py:162-165).
    func testExecutedGoalBlendsWithThePreviousGoal() throws {
        let previous = Array(repeating: Float(0.2), count: joints)
        let descent = JointHandDescent(problems: NoProblems(), updates: 0)
        let outcome = try descent.run(problem(start: 0.1, target: 0, previous: previous))
        for value in outcome.executedGoals[0] {
            XCTAssertEqual(value, 0.35 * 0.1 + 0.65 * 0.2, accuracy: 1e-6)
        }
        XCTAssertEqual(outcome.costs.count, 1)
    }

    /// Without a previous goal the executed goal is P(P(g)); with one it is
    /// P(0.35 P(g) + 0.65 p) (hand_response.py:162-165). With P halving: start 0.4 gives 0.1;
    /// with p = 0.3 it gives 0.5 x (0.35 x 0.2 + 0.65 x 0.3) = 0.1325.
    func testProjectionIsAppliedBeforeAndAfterTheBlend() throws {
        let descent = JointHandDescent(problems: NoProblems(), updates: 0)
        let base = problem(start: 0.4, target: 0)
        func halving(previous: [Float]?) -> HandDescentProblem {
            HandDescentProblem(
                bodies: 1, rawInitialGoal: base.rawInitialGoal, previousGoal: previous,
                lowerLimits: base.lowerLimits, upperLimits: base.upperLimits, valid: [true],
                cost: base.cost, projection: HalvingProjection())
        }
        for value in try descent.run(halving(previous: nil)).executedGoals[0] {
            XCTAssertEqual(value, 0.1, accuracy: 1e-6)
        }
        let previous = Array(repeating: Float(0.3), count: joints)
        for value in try descent.run(halving(previous: previous)).executedGoals[0] {
            XCTAssertEqual(value, 0.5 * (0.35 * 0.2 + 0.65 * 0.3), accuracy: 1e-6)
        }
    }

    /// An invalid body whose cost is NaN scores 0 and does not stop the descent
    /// (hand_response.py:171 masks it with torch.where before the finiteness check at :184).
    func testInvalidBodyWithANonFiniteCostDoesNotThrow() throws {
        let base = problem(bodies: 2, target: 0.05, valid: [false, true])
        let cost = DistancePlusAddendCost(
            target: MLXArray(Array(repeating: Float(0.05), count: 2 * joints), [2, joints]),
            addend: MLXArray([Float.nan, 0], [2]))
        let nanProblem = HandDescentProblem(
            bodies: 2, rawInitialGoal: base.rawInitialGoal, previousGoal: nil,
            lowerLimits: base.lowerLimits, upperLimits: base.upperLimits, valid: [false, true],
            cost: cost, projection: IdentityProjection())
        let outcome = try JointHandDescent(problems: NoProblems()).run(nanProblem)
        XCTAssertEqual(outcome.costs.count, 21)
        XCTAssertTrue(outcome.costs.allSatisfy { $0[0] == 0 })
        XCTAssertEqual(outcome.selectedIteration[0], 0)
        XCTAssertGreaterThan(outcome.selectedIteration[1], 0)
    }

    /// A problem with 0 bodies and a body index outside the outcome throw.
    func testNoBodiesAndBodyOutOfRangeThrow() throws {
        let base = problem(target: 0)
        let empty = HandDescentProblem(
            bodies: 0, rawInitialGoal: [], previousGoal: nil, lowerLimits: base.lowerLimits,
            upperLimits: base.upperLimits, valid: [], cost: base.cost, projection: base.projection)
        let descent = JointHandDescent(problems: NoProblems(), updates: 0)
        XCTAssertThrowsError(try descent.run(empty)) { error in
            XCTAssertEqual(error as? HandDescentError, .noBodies)
        }
        let outcome = try descent.run(base)
        XCTAssertNoThrow(try outcome.result(body: 0))
        XCTAssertThrowsError(try outcome.result(body: 1)) { error in
            XCTAssertEqual(error as? HandDescentError, .bodyOutOfRange(body: 1, bodies: 1))
        }
    }

    /// The linear response with coefficients (0, 1, 0) reaches the goal; an intercept of 0.1
    /// adds 0.1 x range; the cost type measures the predicted angles.
    func testLinearHandResponse() throws {
        let ranges = Array(repeating: Float(2), count: joints)
        var reach: [Float] = []
        var offset: [Float] = []
        for _ in 0 ..< joints {
            reach.append(contentsOf: [0, 1, 0])
            offset.append(contentsOf: [0.1, 0, 0])
        }
        let initial = MLXArray(Array(repeating: Float(0.5), count: joints), [1, joints])
        let velocity = MLXArray(Array(repeating: Float(0), count: joints), [1, joints])
        let goal = MLXArray(Array(repeating: Float(0.8), count: joints), [1, joints])
        let reaching = try LinearHandResponse(coefficients: reach, ranges: ranges)
        for value in reaching.predictedJoints(initial: initial, velocity: velocity, goal: goal, projection: IdentityProjection()).asArray(Float.self) {
            XCTAssertEqual(value, 0.8, accuracy: 1e-6)
        }
        let drifting = try LinearHandResponse(coefficients: offset, ranges: ranges)
        for value in drifting.predictedJoints(initial: initial, velocity: velocity, goal: goal, projection: IdentityProjection()).asArray(Float.self) {
            XCTAssertEqual(value, 0.5 + 0.1 * 2, accuracy: 1e-6)
        }
        let cost = try CalibratedHandGoalCost(
            response: reaching,
            currentAngles: Array(repeating: 0.5, count: joints),
            currentVelocities: Array(repeating: 0, count: joints),
            bodies: 1,
            projection: IdentityProjection(),
            poseError: DistanceError(target: goal))
        XCTAssertEqual(cost.cost(executedGoal: goal).asArray(Float.self)[0], 0, accuracy: 1e-9)
        XCTAssertThrowsError(try LinearHandResponse(coefficients: [0], ranges: ranges))
    }

    /// Wrong counts and a smoothing outside 0 ... 1 throw.
    func testInvalidProblemsThrow() {
        let descent = JointHandDescent(problems: NoProblems())
        var bad = problem(target: 0)
        bad = HandDescentProblem(
            bodies: 1, rawInitialGoal: [0], previousGoal: nil, lowerLimits: bad.lowerLimits,
            upperLimits: bad.upperLimits, valid: [true], cost: bad.cost, projection: bad.projection)
        XCTAssertThrowsError(try descent.run(bad))
        let base = problem(target: 0)
        let smoothing = HandDescentProblem(
            bodies: 1, rawInitialGoal: base.rawInitialGoal, previousGoal: nil, smoothing: 1.5,
            lowerLimits: base.lowerLimits, upperLimits: base.upperLimits, valid: [true],
            cost: base.cost, projection: base.projection)
        XCTAssertThrowsError(try descent.run(smoothing)) { error in
            XCTAssertEqual(error as? HandDescentError, .invalidSmoothing(1.5))
        }
    }
}

/// A problem source with no problems: `descend` returns nil.
private final class NoProblems: HandDescentProblemSource {
    func problem(after command: Command, goal: GoalRingEntry, at step: ModelStep) throws -> HandDescentProblem? { nil }
}
