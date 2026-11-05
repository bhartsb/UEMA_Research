// PlannerViewModel.swift: the plan panel's numbers: the goal ring's fill and the last
// planning event's scores. Work item WI-32, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html. The work item's main file and
// its "First build on a Mac" checklist are RunViewModel.swift.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1).
//
// WHAT IT PUBLISHES after every model step:
//   - the goal ring's fill: entries 0 to 200 (the memory of the last 200 model steps the
//     planner picks its goal from; ModelStep.goalRingLength);
//   - from the last planning event that ran (PlanResult): the score of the start and of
//     each Adam update (21 numbers for a full event: the sum over five senses of the mean
//     squared difference between predicted and goal tokens, plus 0.01 x the sum of
//     squared strengths; planner_loss.md section 3 steps 5 to 8), the index of the
//     selected evaluation and its score, the model step and wellness of the goal ring
//     entry used as the goal, and whether the hand descent gave finger joint goals;
//   - the counts of events that ran and of due events skipped (thermal state serious or
//     critical, MemoryViewModel), and the outcome of the last model step.
// The goal's rank in the ring and the hand descent's 20 scores (REQ-B-14) are not in
// PlanResult (Protocols.swift), so this panel does not carry them.
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave:
// - docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html REQ-B-14: "Plan panel: goal ring
//   fill (0 to 200), the goal's wellness and rank, the last planner event's 21 scores with
//   the selected iteration, the command as a 26 x 9 table, the hand descent's 20 scores."
// - plan row WI-38: "PlannerView: the 21 scores of the last planning event."
// - apple/UEMAKit/Sources/UEMACore/Protocols.swift: PlanResult (scores, selectedIteration,
//   selectedScore, goalStep, goalWellness, fingerGoals; evaluationsPerEvent 21).
// - apple/UEMAKit/Specs/planner_loss.md sections 1 and 3.

import Combine
import Foundation
import UEMACore

/// Publishes the goal ring's fill and the last planning event's scores.
@MainActor
public final class PlannerViewModel: ObservableObject, ModelStepObserving {
    /// The goal ring's entries after the last model step, 0 to 200.
    @Published public private(set) var goalRingCount = 0

    /// The goal ring's capacity: 200 entries.
    public let goalRingCapacity = ModelStep.goalRingLength

    /// The last event's score per evaluation, start first; empty before the first event.
    @Published public private(set) var scores: [Float] = []

    /// The index into `scores` of the selected evaluation; nil before the first event.
    @Published public private(set) var selectedIteration: Int? = nil

    /// The selected evaluation's score; nil before the first event.
    @Published public private(set) var selectedScore: Float? = nil

    /// The model step of the goal ring entry used as the goal; nil before the first event.
    @Published public private(set) var goalStep: ModelStep? = nil

    /// The wellness of that entry; nil before the first event.
    @Published public private(set) var goalWellness: Float? = nil

    /// True when the last event's hand descent gave 44 finger joint goals.
    @Published public private(set) var lastEventHadFingerGoals = false

    /// The model step of the last event that ran; nil before the first.
    @Published public private(set) var lastEventStep: ModelStep? = nil

    /// Planning events that ran since start or reset.
    @Published public private(set) var eventsRun = 0

    /// Due planning events skipped since start or reset.
    @Published public private(set) var eventsSkipped = 0

    /// The last model step's planning outcome; nil before the first model step.
    @Published public private(set) var lastOutcome: PlanningEventOutcome? = nil

    /// An empty plan panel.
    public init() {}

    /// The goal ring's fill as a fraction, 0 (empty) to 1 (200 entries).
    public var goalRingFill: Double {
        Double(goalRingCount) / Double(goalRingCapacity)
    }

    /// Takes the ring count and, when an event ran, its result.
    public func observe(_ report: ModelStepReport) {
        goalRingCount = report.goalRingCount
        lastOutcome = report.planningEvent
        switch report.planningEvent {
        case .ran(let plan):
            scores = plan.scores
            selectedIteration = plan.selectedIteration
            selectedScore = plan.selectedScore
            goalStep = plan.goalStep
            goalWellness = plan.goalWellness
            lastEventHadFingerGoals = plan.fingerGoals != nil
            lastEventStep = report.step
            eventsRun += 1
        case .skipped:
            eventsSkipped += 1
        case .notDue:
            break
        }
    }

    /// Empties the panel.
    public func reset() {
        goalRingCount = 0
        scores = []
        selectedIteration = nil
        selectedScore = nil
        goalStep = nil
        goalWellness = nil
        lastEventHadFingerGoals = false
        lastEventStep = nil
        eventsRun = 0
        eventsSkipped = 0
        lastOutcome = nil
    }
}
