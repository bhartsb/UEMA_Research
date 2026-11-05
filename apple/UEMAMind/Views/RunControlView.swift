// RunControlView.swift: the run controls of UEMA Mind: start, pause, one model step, and
// the model-step counter. Work item WI-38, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html (row WI-38: "RunControlView:
// start, pause, one model step, counter"); specification REQ-B-14: "play, pause, single
// model step; the counter".
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). SwiftUI only (plan section 3, fact 4); no UIKit or
// AppKit type (specification T-B-7: "grep 0 UIKit or AppKit views").
//
// INPUT: the model-step count (UEMACore `ModelStep`, apple/UEMAKit/Sources/UEMACore/
// ModelStep.swift:35), whether the loop is running, whether a run can start (false until
// the weights file is loaded), and three closures. OUTPUT: the three closures are called
// on a button press; the view holds no run state of its own (single responsibility: the
// loop lives in RunViewModel, WI-32, and the composition in UEMAMindApp.swift).
//
// The "First build on a Mac" checklist of this work item is in UEMAMindApp.swift's header.

import Foundation
import SwiftUI
import UEMACore

/// The counter text of the run controls, kept apart from the view so a test reads it.
/// One model step is 50 ms of simulated time (`ModelStep.milliseconds`,
/// ModelStep.swift:57), so the counter also states the simulated seconds.
struct RunCounterText {
    /// "10 model steps, 0.50 s of simulated time (50 ms each)" for a count of 10;
    /// "1 model step, ..." for a count of 1.
    static func line(for step: ModelStep) -> String {
        let noun = step.count == 1 ? "model step" : "model steps"
        let seconds = String(format: "%.2f", step.simulatedSeconds)
        return "\(step.count) \(noun), \(seconds) s of simulated time (\(ModelStep.milliseconds) ms each)"
    }

    /// The count of the next model step whose count is a multiple of 10
    /// (`ModelStep.planningInterval`, lem_model.py:4283 through ModelStep.swift:63-65),
    /// after `step` model steps completed. The next model step to run has count
    /// `step.count` (the first model step has count 0, RunViewModel.swift's
    /// `ModelStep(count: completedModelSteps)`), so the answer is the smallest multiple of
    /// 10 at or above `step.count`: 0 after 0, 10 after 1 to 10, 20 after 11 to 20. A
    /// planning event also needs 20 goal ring entries (ModelStep.swift:70-71), which the
    /// planner panel shows; with one entry recorded per model step the first event runs at
    /// model step 20.
    static func nextPlanningCandidate(after step: ModelStep) -> Int {
        let interval = ModelStep.planningInterval
        return (step.count + interval - 1) / interval * interval
    }
}

/// Start, pause and one-model-step buttons with the model-step counter. A pure
/// presentation of the values it is given: pressing a button calls the matching closure.
struct RunControlView: View {
    /// The model steps completed since the session started.
    let modelStep: ModelStep

    /// True while the loop runs model steps one after another.
    let isRunning: Bool

    /// False until a session exists (the weights file loaded); every button is disabled.
    let canRun: Bool

    /// Called by "Start".
    let onStart: () -> Void

    /// Called by "Pause".
    let onPause: () -> Void

    /// Called by "One model step": exactly one model step (10 physics steps of 5 ms).
    let onStepOnce: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Button("Start", action: onStart)
                    .disabled(!canRun || isRunning)
                Button("Pause", action: onPause)
                    .disabled(!canRun || !isRunning)
                Button("One model step", action: onStepOnce)
                    .disabled(!canRun || isRunning)
            }
            Text(RunCounterText.line(for: modelStep))
                .font(.body.monospacedDigit())
                .accessibilityIdentifier("modelStepCounter")
            Text("Next model step whose count is a multiple of \(ModelStep.planningInterval) (a planning candidate): \(RunCounterText.nextPlanningCandidate(after: modelStep))")
                .font(.caption)
                .foregroundStyle(.secondary)
            if !canRun {
                Text("Pick model.safetensors to load the weights before the first model step.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

#Preview("Run controls, 10 model steps") {
    RunControlView(
        modelStep: ModelStep(count: 10),
        isRunning: false,
        canRun: true,
        onStart: {},
        onPause: {},
        onStepOnce: {})
        .padding()
}
