// PlannerTestSupport.swift: the exact scalar forward model of
// Source/Python/lem/tests/test_planner_selection.py (lines 50-60: "value = action[:, 0, :1]",
// a goal of 0) as a PlanTransition, and helpers shared by the planner tests (work item WI-39).
// Nothing in this file was compiled on the machine that wrote it (no Swift toolchain on
// Windows).

import MLX
import UEMACore
@testable import UEMAPlanner

/// Predicts every one of the 192 numbers of the tactile token as the command's number at
/// `index` (0 = handle 0's position x), through a [234, 192] selection matrix, so that the
/// goal term against a zero goal token is exactly that number squared.
final class ScalarTransition: PlanTransition {
    let scoredSenses: [SenseName]
    private let selector: MLXArray

    init(index: Int = 0, scoredSenses: [SenseName] = [.tactile]) {
        self.scoredSenses = scoredSenses
        var values = Array(repeating: Float(0), count: Command.count * Token.count)
        for column in 0 ..< Token.count {
            values[index * Token.count + column] = 1
        }
        selector = MLXArray(values, [Command.count, Token.count])
    }

    func predictNext(rawCommands: MLXArray) -> [SenseName: MLXArray] {
        let bodies = rawCommands.shape[0]
        let flat = rawCommands.reshaped([bodies, Command.count])
        let predicted = MLX.matmul(flat, selector)
        return [.tactile: predicted, .vision: predicted]
    }
}

/// Predicts the tactile token from a closure of the raw commands [B, 26, 9] and the call
/// count (1 for the first call): test_planner_selection.py's `scalar_transition(monkeypatch,
/// fn)` pattern. The closure returns [B, 1]; the 192 numbers of the token all equal it.
final class ClosureTransition: PlanTransition {
    let scoredSenses: [SenseName] = [.tactile]
    private let value: (MLXArray, Int) -> MLXArray
    private(set) var calls = 0

    init(_ value: @escaping (MLXArray, Int) -> MLXArray) { self.value = value }

    func predictNext(rawCommands: MLXArray) -> [SenseName: MLXArray] {
        calls += 1
        let bodies = rawCommands.shape[0]
        let column = value(rawCommands, calls).reshaped([bodies, 1])
        return [.tactile: MLX.matmul(column, MLXArray(Array(repeating: Float(1), count: Token.count), [1, Token.count]))]
    }
}

/// Hands out one transition for every model step.
final class FixedTransitionProvider: PlanTransitionProviding {
    let transition: any PlanTransition
    init(_ transition: any PlanTransition) { self.transition = transition }
    func transition(at step: ModelStep) throws -> any PlanTransition { transition }
}

/// test_planner_selection.py's motor_at: handle 0's position x = `x`, every quaternion
/// (0, 0, 0, 1), radius and strength 0.5.
func commandAt(_ x: Float) -> Command {
    var command = Command.zero(scale: .normalized)
    for handle in BodyHandle.allCases {
        command[handle, .quaternionW] = 1
        command[handle, .radius] = 0.5
        command[handle, .strength] = 0.5
    }
    command[.wristL, .positionX] = x
    return command
}

/// The goal of test_planner_selection.py: every token 0.
let zeroGoal: TokenSet = [.tactile: Token.zero, .vision: Token.zero]
