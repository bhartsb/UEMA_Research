// PlannerView.swift: the planner panel of UEMA Mind: goal ring fill, the goal's model step
// and wellness, and the 21 scores of the last planning event with the selected one. Work
// item WI-38, target B of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html (row
// WI-38: "PlannerView: the 21 scores of the last planning event"); specification
// REQ-B-14: "Plan panel: goal ring fill (0 to 200), the goal's wellness and rank, the last
// planner event's 21 scores with the selected iteration". The goal's rank and the hand
// descent's 20 scores are not in UEMACore's `PlanResult` (Protocols.swift:692-775), so
// this panel does not show them.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). SwiftUI only; no UIKit or AppKit type.
//
// INPUT: the numbers PlannerViewModel publishes after each model step
// (apple/UEMAMind/ViewModels/PlannerViewModel.swift:36-117, WI-32): the goal ring's
// entries (0 to 200, `ModelStep.goalRingLength`, ModelStep.swift:67-68; a planning event
// needs 20, :70-71); the last event's `scores`, the total score per evaluation, start
// first ("`parts[:, 0]` per iteration", planner_loss.md section 6 point D), 21 for a full
// event (the start and 20 Adam updates at step size 0.1, Protocols.swift:699-706), fewer
// when a non-finite score stopped the search; the index of the selected evaluation; the
// goal ring entry used as the goal (its model step and wellness); the model step of the
// last event; and the count of due events skipped (WI-32: a serious or critical thermal
// state skips the event). OPERATION: `PlannerScoreBars.make` gives each score a bar
// length: the score minus the smallest score of the event, over the largest minus the
// smallest (0 = the event's lowest score, 1 = its highest; lower scores are better, since
// the score is a loss: planner_loss.md section 3). OUTPUT: pixels; nothing is written
// back.
//
// The "First build on a Mac" checklist of this work item is in UEMAMindApp.swift's header.

import Foundation
import SwiftUI
import UEMACore

/// One bar of the score chart.
struct PlannerScoreBar: Equatable, Identifiable {
    /// 0 for the start, 1 to 20 for the Adam updates.
    let iteration: Int

    /// The total score of this evaluation (lower is better).
    let score: Float

    /// The bar length, 0 (the event's lowest score) to 1 (its highest); 0 for every bar of
    /// an event whose scores are all equal, and for a non-finite score.
    let fraction: Double

    /// True for the evaluation the planner selected.
    let isSelected: Bool

    var id: Int { iteration }
}

/// Turns an event's scores into bars. One responsibility: the chart's arithmetic.
enum PlannerScoreBars {
    /// One bar per score, start first; `selectedIteration` nil marks no bar.
    static func make(scores: [Float], selectedIteration: Int?) -> [PlannerScoreBar] {
        let finite = scores.filter(\.isFinite)
        let low = finite.min() ?? 0
        let high = finite.max() ?? 0
        let span = high - low
        return scores.enumerated().map { index, score in
            let fraction: Double
            if score.isFinite, span > 0 {
                fraction = Double((score - low) / span)
            } else {
                fraction = 0
            }
            return PlannerScoreBar(
                iteration: index,
                score: score,
                fraction: fraction,
                isSelected: index == selectedIteration)
        }
    }

    /// "selected update 14 of 20, score 0.8123 against the start's 1.204" style summary;
    /// "selected the start (score 1.204) of 21 evaluations" when the start was selected;
    /// "no scores" for an empty list or an index outside it.
    static func summary(scores: [Float], selectedIteration: Int?) -> String {
        guard let selected = selectedIteration, scores.indices.contains(selected) else {
            return "no scores"
        }
        let start = String(format: "%.4g", Double(scores[0]))
        let chosen = String(format: "%.4g", Double(scores[selected]))
        let updates = scores.count - 1
        if selected == 0 {
            return "selected the start (score \(start)) of \(scores.count) evaluations"
        }
        return "selected update \(selected) of \(updates), score \(chosen) against the start's \(start)"
    }
}

/// The planner panel.
struct PlannerView: View {
    /// The goal ring's entries now, 0 to 200.
    let goalRingCount: Int

    /// The last event's score per evaluation, start first; empty before the first event.
    let scores: [Float]

    /// The index into `scores` of the selected evaluation; nil before the first event.
    let selectedIteration: Int?

    /// The model step of the goal ring entry used as the goal; nil before the first event.
    let goalStep: ModelStep?

    /// The wellness of that entry; nil before the first event.
    let goalWellness: Float?

    /// The model step of the last event that ran; nil before the first.
    let lastEventStep: ModelStep?

    /// Due planning events skipped since the session started (thermal state serious or
    /// critical).
    let eventsSkipped: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Planner").font(.headline)
            ProgressView(
                value: Double(min(goalRingCount, ModelStep.goalRingLength)),
                total: Double(ModelStep.goalRingLength))
            Text("Goal ring: \(goalRingCount) of \(ModelStep.goalRingLength) model steps held; planning needs \(ModelStep.goalRingMinimumForPlanning) and runs on model steps whose count is a multiple of \(ModelStep.planningInterval)")
                .font(.caption)
            if eventsSkipped > 0 {
                Text("Due planning events skipped (thermal state serious or critical): \(eventsSkipped)")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            if let lastEventStep, !scores.isEmpty {
                Text("Last planning event at model step \(lastEventStep.count): \(goalText)")
                    .font(.caption)
                Text("\(scores.count) scores (\(PlanResult.evaluationsPerEvent) for a full event); \(PlannerScoreBars.summary(scores: scores, selectedIteration: selectedIteration))")
                    .font(.caption)
                scoreChart(PlannerScoreBars.make(scores: scores, selectedIteration: selectedIteration))
            } else {
                Text("No planning event yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// "goal = the goal ring entry of model step 7, wellness 0.42".
    private var goalText: String {
        let step = goalStep.map { String($0.count) } ?? "none"
        let wellness = goalWellness.map { String(format: "%.4g", Double($0)) } ?? "none"
        return "goal = the goal ring entry of model step \(step), wellness \(wellness)"
    }

    private func scoreChart(_ bars: [PlannerScoreBar]) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(bars) { bar in
                HStack(spacing: 6) {
                    Text(bar.iteration == 0 ? "start" : "\(bar.iteration)")
                        .frame(width: 36, alignment: .trailing)
                    GeometryReader { proxy in
                        Rectangle()
                            .fill(bar.isSelected ? Color.accentColor : Color.gray.opacity(0.5))
                            .frame(width: max(2, proxy.size.width * CGFloat(bar.fraction)))
                    }
                    .frame(height: 10)
                    Text(String(format: "%.4g", Double(bar.score)))
                        .frame(width: 72, alignment: .trailing)
                }
                .font(.caption2.monospacedDigit())
            }
            Text("Bar length: 0 = the event's lowest score, full = its highest; the selected evaluation is coloured.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}

#if DEBUG
/// Synthetic scores for the preview: falling from 1.2 by 0.03 per update, the last one
/// selected. Not a planner's output.
enum PlannerPreviewData {
    static let scores: [Float] = (0 ..< PlanResult.evaluationsPerEvent).map { Float(1.2 - 0.03 * Double($0)) }
}

#Preview("Planner, synthetic event") {
    ScrollView {
        PlannerView(
            goalRingCount: 40,
            scores: PlannerPreviewData.scores,
            selectedIteration: PlannerPreviewData.scores.count - 1,
            goalStep: ModelStep(count: 7),
            goalWellness: 0.42,
            lastEventStep: ModelStep(count: 40),
            eventsSkipped: 0)
            .padding()
    }
}
#endif
