// AdamCommandSearchTests.swift: the command search on the exact scalar forward model of
// Source/Python/lem/tests/test_planner_selection.py, and the Adam update (work item WI-39).
// Nothing in this file was compiled on the machine that wrote it (no Swift toolchain on
// Windows). Every pass line is the Python test's or follows from torch.optim.Adam's update.

import XCTest
import MLX
import UEMACore
@testable import UEMAPlanner

final class TorchAdamTests: XCTestCase {
    /// The first update moves a parameter by the step size against the gradient's sign:
    /// 1.0 with gradient 2.0 becomes 0.9 (0.9000000005 in float64 arithmetic).
    func testFirstUpdateMovesByTheStepSize() {
        var adam = TorchAdam(learningRate: 0.1, parameterCounts: [1])
        var parameters: [[Float]] = [[1.0]]
        adam.update(&parameters, gradients: [[2.0]])
        XCTAssertEqual(parameters[0][0], 0.9, accuracy: 1e-6)
        XCTAssertEqual(adam.stepCount, 1)
    }

    /// Two updates with gradients 2.0 then 1.0 give 0.80678204 (float64 arithmetic of the
    /// torch formula: m 0.28, v 0.004996, step 0.1 / 0.19, denominator 1.58090).
    func testSecondUpdateUsesBiasCorrectedMoments() {
        var adam = TorchAdam(learningRate: 0.1, parameterCounts: [1])
        var parameters: [[Float]] = [[1.0]]
        adam.update(&parameters, gradients: [[2.0]])
        adam.update(&parameters, gradients: [[1.0]])
        XCTAssertEqual(parameters[0][0], 0.80678204, accuracy: 1e-5)
    }

    /// A zero gradient leaves a parameter where it is.
    func testZeroGradientLeavesTheParameter() {
        var adam = TorchAdam(learningRate: 0.1, parameterCounts: [2])
        var parameters: [[Float]] = [[0.3, -0.7]]
        adam.update(&parameters, gradients: [[0, 0]])
        XCTAssertEqual(parameters[0], [0.3, -0.7])
    }
}

final class AdamCommandSearchTests: XCTestCase {
    private func search(updates: Int = 20, strength: Float = 0) -> AdamCommandSearch {
        AdamCommandSearch(updates: updates, learningRate: 0.1, strengthRegularisation: strength)
    }

    /// test_twenty_adam_updates_cannot_discard_a_better_initial_command: from x = 0.001 the
    /// chosen command scores at or below the start and equals the minimum of the scores.
    func testTwentyUpdatesCannotDiscardABetterStart() throws {
        let start = commandAt(0.001)
        let result = try search().run(start: [start], goals: [zeroGoal], transition: ScalarTransition())
        let chosenX = result.commands[0][0]
        XCTAssertLessThanOrEqual(chosenX * chosenX, Float(0.001) * Float(0.001) + 1e-12)
        let totals = result.totals(body: 0)
        let minimum = try XCTUnwrap(totals.min())
        XCTAssertEqual(result.selectedTotal(body: 0), minimum, accuracy: 1e-12)
        XCTAssertEqual(result.selectedTotal(body: 0), chosenX * chosenX, accuracy: 1e-9)
    }

    /// The released settings evaluate 21 times and update 20 times (PlanResult constants).
    func testTwentyOneEvaluationsAndTwentyUpdates() throws {
        let result = try search().run(start: [commandAt(0.5)], goals: [zeroGoal], transition: ScalarTransition())
        XCTAssertEqual(result.evaluations.count, PlanResult.evaluationsPerEvent)
        XCTAssertEqual(result.updates, PlanResult.updatesPerEvent)
        XCTAssertEqual(result.stoppedBy, .stepBudget)
        XCTAssertLessThanOrEqual(result.selectedTotal(body: 0), result.evaluations[0][0].total)
    }

    /// test_each_body_keeps_its_own_best_candidate_and_final_update_is_evaluated: with one
    /// update, a body at 0.001 keeps its start and a body at 1.0 takes the update (0.9).
    func testEachBodyKeepsItsOwnBestAndTheFinalUpdateIsEvaluated() throws {
        let result = try search(updates: 1).run(
            start: [commandAt(0.001), commandAt(1.0)], goals: [zeroGoal, zeroGoal], transition: ScalarTransition())
        XCTAssertEqual(result.evaluations.count, 2)
        XCTAssertEqual(result.commands[0][0], 0.001, accuracy: 1e-7)
        XCTAssertEqual(result.selectedIteration[0], 0)
        XCTAssertEqual(result.commands[1][0], 0.9, accuracy: 1e-6)
        XCTAssertEqual(result.selectedIteration[1], 1)
    }

    /// The gradient reaches only the number the score depends on: after one update from
    /// x = 1.0 every other one of the 234 numbers equals the start. A gradient that also took
    /// in the search's auxiliary outputs (the scores and the command) would move them.
    func testGradientStepMatchesAdamOnTheScalarModel() throws {
        let start = commandAt(1.0)
        let result = try search(updates: 1).run(start: [start], goals: [zeroGoal], transition: ScalarTransition())
        XCTAssertEqual(result.selectedIteration[0], 1)
        XCTAssertEqual(result.commands[0][0], 0.9, accuracy: 1e-6)
        for index in 1 ..< Command.count {
            XCTAssertEqual(result.commands[0][index], start.values[index], accuracy: 1e-6, "number \(index)")
        }
    }

    /// With no scored sense the score is the effort alone: 0.01 x 26 handles x 0.5^2 = 0.065
    /// at the start, and the search lowers it.
    func testEffortIsOneHundredthOfTheSquaredStrengths() throws {
        let result = try search(strength: 0.01).run(
            start: [commandAt(0)], goals: [zeroGoal], transition: ScalarTransition(scoredSenses: []))
        XCTAssertEqual(result.evaluations[0][0].effort, 0.065, accuracy: 1e-6)
        XCTAssertEqual(result.evaluations[0][0].goal, 0, accuracy: 1e-12)
        XCTAssertLessThan(result.selectedTotal(body: 0), 0.065)
    }

    /// test_selected_score_includes_the_effort_tradeoff: the tactile prediction is handle 0's
    /// strength minus 0.5 and the strength weight is 1.0. The start scores goal 0 and effort
    /// 26 handles x 0.5^2 = 6.5; the selected command lowers the total below 6.5 while its
    /// goal term rises above 0, and its total is its goal plus its effort.
    func testSelectedScoreIncludesTheEffortTradeoff() throws {
        let transition = ClosureTransition { commands, _ in
            commands.reshaped([commands.shape[0], Command.count])[0..., 8 ..< 9] - MLXArray(Float(0.5))
        }
        let result = try search(strength: 1.0).run(start: [commandAt(0)], goals: [zeroGoal], transition: transition)
        XCTAssertEqual(result.evaluations[0][0].effort, 6.5, accuracy: 1e-5)
        XCTAssertEqual(result.evaluations[0][0].goal, 0, accuracy: 1e-12)
        let selected = result.evaluations[result.selectedIteration[0]][0]
        XCTAssertGreaterThan(selected.goal, 0)
        XCTAssertLessThan(selected.total, 6.5)
        XCTAssertEqual(selected.total, selected.goal + selected.effort, accuracy: 1e-5)
    }

    /// test_later_nonfinite_candidate_cannot_replace_finite_initial_command: after the start,
    /// every prediction is NaN; the search stops at the first NaN evaluation and keeps the
    /// start command.
    func testLaterNonFiniteCandidateCannotReplaceTheStart() throws {
        let transition = ClosureTransition { commands, call in
            let x = commands.reshaped([commands.shape[0], Command.count])[0..., 0 ..< 1]
            return call == 1 ? x : x * MLXArray(Float.nan)
        }
        let result = try search(updates: 2).run(start: [commandAt(0.001)], goals: [zeroGoal], transition: transition)
        XCTAssertEqual(result.stoppedBy, .nonfiniteCandidate)
        XCTAssertEqual(result.evaluations.count, 2)
        XCTAssertEqual(result.selectedIteration[0], 0)
        XCTAssertTrue(result.commands[0].allSatisfy { $0.isFinite })
        XCTAssertEqual(result.commands[0][0], 0.001, accuracy: 1e-9)
    }

    /// test_no_finite_candidate_raises_instead_of_issuing_an_unscored_command.
    func testNoFiniteCandidateThrows() {
        let transition = ClosureTransition { commands, _ in
            commands.reshaped([commands.shape[0], Command.count])[0..., 0 ..< 1] * MLXArray(Float.nan)
        }
        XCTAssertThrowsError(
            try search(updates: 2).run(start: [commandAt(0.001)], goals: [zeroGoal], transition: transition)
        ) { error in
            XCTAssertEqual(error as? PlannerError, .nonFiniteStart)
        }
    }

    /// A non-finite start throws (intention_generator.py:373-375).
    func testNonFiniteStartThrows() {
        XCTAssertThrowsError(
            try search().run(start: [commandAt(.nan)], goals: [zeroGoal], transition: ScalarTransition())
        ) { error in
            XCTAssertEqual(error as? PlannerError, .nonFiniteStart)
        }
    }

    /// The search takes `.normalized` commands only, one goal per command.
    func testWrongScaleAndCountThrow() {
        let centimetres = Command.zero(scale: .centimetres)
        XCTAssertThrowsError(
            try search().run(start: [centimetres], goals: [zeroGoal], transition: ScalarTransition())
        ) { error in
            XCTAssertEqual(error as? PlannerError, .wrongScale(found: .centimetres))
        }
        XCTAssertThrowsError(
            try search().run(start: [commandAt(0)], goals: [], transition: ScalarTransition())
        ) { error in
            XCTAssertEqual(error as? PlannerError, .bodyCountMismatch(commands: 1, goals: 0))
        }
    }

    /// The direct parameterization returns the start command unchanged before any update.
    func testParameterizationRoundTripsTheStart() throws {
        let start = commandAt(0.25)
        let parameterization = DirectCommandParameterization()
        let parameters = try parameterization.initialParameters(from: start.values, bodies: 1)
        let rebuilt = parameterization.assemble(parameters).command.asArray(Float.self)
        XCTAssertEqual(rebuilt.count, Command.count)
        for index in 0 ..< Command.count {
            XCTAssertEqual(rebuilt[index], start.values[index], accuracy: 1e-6, "number \(index)")
        }
    }
}
