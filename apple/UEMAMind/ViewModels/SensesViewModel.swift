// SensesViewModel.swift: the sense panel's numbers for each of the six active senses.
// Work item WI-32, target B of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html.
// The work item's main file and its "First build on a Mac" checklist are RunViewModel.swift.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1).
//
// WHAT IT COMPUTES, per active sense, from each model step's ModelStepReport:
//   - the encoder input's shape (primary, and auxiliary when present), its minimum and
//     maximum over the primary values (nil for an empty array);
//   - the sense's token (192 numbers) and the same 192 numbers as a heat map of 12 rows x
//     16 columns, row r holding token numbers 16 r to 16 r + 15;
//   - the prediction error: the mean over 192 of (predicted - actual)^2 between the token
//     the prediction heads gave for this sense at the previous model step and this model
//     step's token (Token.meanSquaredDifference); nil on the first model step after start
//     or reset, which has no previous prediction.
//   - whether the reading was fresh (vision is not fresh on a model step without a render,
//     planner_loss.md section 3 step 5).
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave:
// - docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html REQ-B-14: "Sense panel: per sense
//   the input's shape, minimum and maximum, the token as a 12 x 16 heat map, the
//   prediction error (mean squared difference)."
// - plan row WI-38: "SensesView: six tokens as 192-cell strips."
// - apple/UEMAKit/Sources/UEMACore/SenseName.swift: SenseName.active (vision, tactile,
//   audio, proprioception, motor, vestibular), Token.count 192,
//   Token.meanSquaredDifference(to:).
// - apple/UEMAKit/Sources/UEMACore/Protocols.swift: ModelOutput.predictedTokens "The
//   prediction heads' tokens for the next model step, one per active sense".

import Combine
import Foundation
import UEMACore

/// The sense panel's numbers for one active sense at one model step.
public struct SensePanelRow: Identifiable, Equatable {
    /// Heat map rows: 12.
    public static let heatMapRows = 12

    /// Heat map columns: 16 (12 x 16 = 192 = Token.count).
    public static let heatMapColumns = 16

    /// The sense.
    public let sense: SenseName

    /// The sense, as the row's identity.
    public var id: SenseName { sense }

    /// The encoder input's primary shape.
    public let inputShape: [Int]

    /// The encoder input's auxiliary shape; nil when the sense has none.
    public let auxiliaryShape: [Int]?

    /// The smallest primary value; nil for an empty array.
    public let inputMinimum: Float?

    /// The largest primary value; nil for an empty array.
    public let inputMaximum: Float?

    /// True when the reading was taken this model step.
    public let isFresh: Bool

    /// The sense's token (192 numbers).
    public let token: Token

    /// The token as 12 rows of 16 numbers.
    public let heatMap: [[Float]]

    /// The mean squared difference between the previous model step's predicted token and
    /// this token; nil on the first model step.
    public let predictionError: Float?

    /// The 192 numbers of `token` as 12 rows x 16 columns, row r holding numbers 16 r to
    /// 16 r + 15.
    public static func heatMap(of token: Token) -> [[Float]] {
        (0 ..< heatMapRows).map { row in
            let start = row * heatMapColumns
            return Array(token.values[start ..< start + heatMapColumns])
        }
    }
}

/// Publishes one SensePanelRow per active sense after every model step.
@MainActor
public final class SensesViewModel: ObservableObject, ModelStepObserving {
    /// The rows of the last model step, in the active sequence (vision first); empty
    /// before the first model step.
    @Published public private(set) var rows: [SensePanelRow] = []

    /// The model step the rows belong to; nil before the first.
    @Published public private(set) var step: ModelStep? = nil

    private var previousPredicted: TokenSet = [:]

    /// An empty sense panel.
    public init() {}

    /// Builds the six rows from `report` and keeps its predicted tokens for the next
    /// model step's prediction error.
    public func observe(_ report: ModelStepReport) {
        var built: [SensePanelRow] = []
        for sense in SenseName.active {
            guard let token = report.output.tokens[sense] else { continue }
            let input = report.inputs[sense]
            let values = input?.primary.values ?? []
            let error = previousPredicted[sense].map { $0.meanSquaredDifference(to: token) }
            built.append(SensePanelRow(
                sense: sense,
                inputShape: input?.primary.shape ?? [],
                auxiliaryShape: input?.auxiliary?.shape,
                inputMinimum: values.min(),
                inputMaximum: values.max(),
                isFresh: input?.isFresh ?? false,
                token: token,
                heatMap: SensePanelRow.heatMap(of: token),
                predictionError: error))
        }
        rows = built
        step = report.step
        previousPredicted = report.output.predictedTokens
    }

    /// Empties the panel and forgets the previous prediction.
    public func reset() {
        rows = []
        step = nil
        previousPredicted = [:]
    }
}
