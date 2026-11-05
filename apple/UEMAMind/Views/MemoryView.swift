// MemoryView.swift: the memory and thermal readout of UEMA Mind. Work item WI-38, target
// B of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html (row WI-38: "MemoryView:
// available memory in MB, thermal state"); specification REQ-B-14: "memory readout
// (os_proc_available_memory)"; REQ-B-15: the peak footprint is "the drop of
// os_proc_available_memory since launch", with a pass line below 3,000 MB on an iPhone
// with the increased-memory-limit entitlement.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). SwiftUI only; no UIKit or AppKit type. The view
// reads no system call itself: MemoryViewModel (WI-32, apple/UEMAMind/ViewModels/
// MemoryViewModel.swift:84-119) reads os_proc_available_memory and
// ProcessInfo.thermalState, divides bytes by 1,048,576 for MB, and keeps the largest drop
// since launch as the peak footprint; this view shows those three numbers.
//
// INPUT: available memory now in MB and the peak footprint since launch in MB (both nil
// on macOS, where os_proc_available_memory does not exist: MemoryViewModel.swift:10-13),
// and `ProcessInfo.ThermalState` (nominal, fair, serious, critical; specification
// REQ-B-16). OPERATION: `MemoryReadout` formats the numbers and names the thermal state
// with its effect on the planner. OUTPUT: pixels; nothing is written back.
//
// The "First build on a Mac" checklist of this work item is in UEMAMindApp.swift's header.

import Foundation
import SwiftUI

/// The readout's wording. One responsibility: turn the view model's numbers into text, kept
/// apart from the view so a test reads it.
enum MemoryReadout {
    /// The specification's peak-footprint line on an iPhone: 3,000 MB (REQ-B-15).
    static let footprintLineMegabytes: Double = 3_000

    /// "1234 MB" with no decimals; "not reported on this platform" for nil.
    static func text(megabytes: Double?) -> String {
        guard let megabytes else { return "not reported on this platform" }
        return String(format: "%.0f MB", megabytes)
    }

    /// The thermal state's name and its effect on the planner (WI-32,
    /// MemoryViewModel.swift:121-125: the planning event is skipped at serious and
    /// critical).
    static func text(thermalState: ProcessInfo.ThermalState) -> String {
        switch thermalState {
        case .nominal:
            return "nominal (planning events run)"
        case .fair:
            return "fair (planning events run)"
        case .serious:
            return "serious (planning events skipped)"
        case .critical:
            return "critical (planning events skipped)"
        @unknown default:
            return "unknown value \(thermalState.rawValue) (planning events run)"
        }
    }
}

/// Available memory, the peak footprint since launch, and the thermal state.
struct MemoryView: View {
    /// os_proc_available_memory now, MB; nil where not reported.
    let availableMegabytes: Double?

    /// The largest drop of the available memory since launch, MB; nil where not reported.
    let peakFootprintMegabytes: Double?

    /// ProcessInfo.thermalState at the last reading.
    let thermalState: ProcessInfo.ThermalState

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Memory and temperature").font(.headline)
            Text("Available memory: \(MemoryReadout.text(megabytes: availableMegabytes))")
            Text("Peak use since launch: \(MemoryReadout.text(megabytes: peakFootprintMegabytes)) (iPhone line: below \(Int(MemoryReadout.footprintLineMegabytes)) MB)")
            Text("Thermal state: \(MemoryReadout.text(thermalState: thermalState))")
        }
        .font(.caption.monospacedDigit())
    }
}

#Preview("Memory, 2,500 MB available") {
    MemoryView(availableMegabytes: 2_500, peakFootprintMegabytes: 400, thermalState: .fair)
        .padding()
}
