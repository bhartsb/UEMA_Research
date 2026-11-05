// PlannerTests.swift: the Planner's cadence, goal choice and result on the exact scalar
// forward model (work item WI-39). Nothing in this file was compiled on the machine that
// wrote it (no Swift toolchain on Windows). Pass lines from planner_loss.md section 1
// ("at least 20 entries and `step % 10 == 0`") and the plan row's acceptance ("the planner
// returns the command it scored best").

import XCTest
import UEMACore
@testable import UEMAPlanner

/// A hand descent that returns one fixed result and counts its calls.
private final class FixedHandDescent: HandDescending {
    private(set) var calls = 0
    let result: HandDescentResult
    init(result: HandDescentResult) { self.result = result }
    func descend(after command: Command, goal: GoalRingEntry, at step: ModelStep) throws -> HandDescentResult? {
        calls += 1
        return result
    }
}

/// A command search that returns a result for 2 bodies when asked for 1.
private final class TwoBodySearch: CommandSearching {
    func run(start: [Command], goals: [TokenSet], transition: any PlanTransition) throws -> CommandSearchResult {
        let parts = ScoreParts(total: 0, goal: 0, effort: 0)
        return CommandSearchResult(
            bodies: 2, commands: [start[0].values, start[0].values], evaluations: [[parts, parts]],
            selectedIteration: [0, 0], updates: 0, stoppedBy: .stepBudget)
    }
}

final class PlannerTests: XCTestCase {
    private func makePlanner(handDescent: (any HandDescending)? = nil) -> Planner {
        Planner(
            ring: GoalRing(),
            goalRule: HighestWellnessGoalSelection(),
            transitions: FixedTransitionProvider(ScalarTransition()),
            search: AdamCommandSearch(strengthRegularisation: 0),
            handDescent: handDescent)
    }

    /// Fills `count` model steps; the entry at model step `bestStep` has the highest wellness
    /// and a zero tactile token, the others a tactile token of 0.3.
    private func fill(_ planner: Planner, count: Int, bestStep: Int) {
        for step in 1 ... count {
            let token = step == bestStep ? Token.zero : Token(repeating: 0.3)
            planner.record(tokens: [.tactile: token], wellness: step == bestStep ? 5 : Float(step) / 100,
                           at: ModelStep(count: step))
        }
    }

    /// A planning event runs on a multiple of 10 once the ring holds 20 entries.
    func testCadenceIsEveryTenthModelStepFromTwentyEntries() {
        let planner = makePlanner()
        fill(planner, count: 19, bestStep: 3)
        XCTAssertFalse(planner.canPlan(at: ModelStep(count: 20)))
        planner.record(tokens: [.tactile: Token.zero], wellness: 0, at: ModelStep(count: 20))
        XCTAssertTrue(planner.canPlan(at: ModelStep(count: 20)))
        XCTAssertFalse(planner.canPlan(at: ModelStep(count: 21)))
        XCTAssertTrue(planner.canPlan(at: ModelStep(count: 30)))
    }

    /// The result carries 21 scores, the selected one at or below the start, the command it
    /// scored best, and the highest-wellness entry as the goal.
    func testPlanReturnsTheBestScoredCommandTowardTheHighestWellnessEntry() throws {
        let planner = makePlanner()
        fill(planner, count: 25, bestStep: 7)
        let result = try planner.plan(from: commandAt(0.6), at: ModelStep(count: 30))
        XCTAssertEqual(result.scores.count, PlanResult.evaluationsPerEvent)
        XCTAssertLessThanOrEqual(result.selectedScore, result.scores[0])
        XCTAssertEqual(result.selectedScore, try XCTUnwrap(result.scores.min()), accuracy: 1e-12)
        let x = result.command[.wristL, .positionX]
        XCTAssertEqual(result.selectedScore, x * x, accuracy: 1e-6)
        XCTAssertEqual(result.goalStep.count, 7)
        XCTAssertEqual(result.goalWellness, 5)
        XCTAssertNil(result.fingerGoals)
        let record = try XCTUnwrap(planner.lastEvent)
        XCTAssertEqual(record.goalIndex, 6)
        XCTAssertEqual(record.goalRank, 1)
        XCTAssertEqual(record.goalRingCount, 25)
    }

    /// The hand descent runs after the search and its 44 goals reach the result.
    func testHandDescentGoalsReachTheResult() throws {
        let goals = try FingerJointGoals(angles: Array(repeating: 0.1, count: FingerJointGoals.count))
        let hand = FixedHandDescent(result: HandDescentResult(
            fingerGoals: goals, rawGoal: goals.angles, costs: Array(repeating: 0, count: 21), selectedIteration: 0))
        let planner = makePlanner(handDescent: hand)
        fill(planner, count: 20, bestStep: 2)
        let result = try planner.plan(from: commandAt(0.2), at: ModelStep(count: 20))
        XCTAssertEqual(hand.calls, 1)
        XCTAssertEqual(result.fingerGoals, goals)
    }

    /// A search result that does not hold exactly 1 body is refused, not indexed.
    func testSearchResultWithTheWrongBodyCountThrows() {
        let planner = Planner(
            ring: GoalRing(), goalRule: HighestWellnessGoalSelection(),
            transitions: FixedTransitionProvider(ScalarTransition()), search: TwoBodySearch(), handDescent: nil)
        fill(planner, count: 20, bestStep: 1)
        XCTAssertThrowsError(try planner.plan(from: commandAt(0), at: ModelStep(count: 20))) { error in
            guard case .badParameters? = error as? PlannerError else {
                return XCTFail("expected PlannerError.badParameters, got \(error)")
            }
        }
    }

    /// An empty ring has no goal; reset empties the ring; a centimetre command is refused.
    func testNoGoalResetAndScale() throws {
        let planner = makePlanner()
        XCTAssertThrowsError(try planner.plan(from: commandAt(0), at: ModelStep(count: 10))) { error in
            XCTAssertEqual(error as? PlannerError, .noGoal)
        }
        fill(planner, count: 20, bestStep: 1)
        XCTAssertThrowsError(try planner.plan(from: Command.zero(scale: .centimetres), at: ModelStep(count: 20))) { error in
            XCTAssertEqual(error as? PlannerError, .wrongScale(found: .centimetres))
        }
        planner.reset()
        XCTAssertEqual(planner.goalRingCount, 0)
        XCTAssertNil(planner.lastEvent)
    }
}
