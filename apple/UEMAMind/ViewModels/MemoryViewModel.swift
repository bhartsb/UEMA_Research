// MemoryViewModel.swift: the memory and thermal readout of UEMA Mind and the rule that
// skips the planner's event at a high thermal state. Work item WI-32, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html. The work item's main file and
// its "First build on a Mac" checklist are RunViewModel.swift.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1).
//
// WHAT IT READS AND WHAT IT DOES WITH IT.
//   Input 1: os_proc_available_memory(), "The number of bytes the app may allocate before
//     it hits its memory limit" (0 when the process is not an app or is already over its
//     limit). It exists on iOS 13.0 and newer and not on macOS (availability "macOS: -"),
//     so on macOS the reading is nil and the screen shows "unavailable".
//   Input 2: ProcessInfo.thermalState: nominal, fair, serious or critical.
//   Output 1: available memory in MB (bytes / 1,048,576), the reading at launch, and the
//     peak footprint: the largest drop of the available memory since launch, in MB, the
//     quantity REQ-B-15 judges ("Peak footprint (the drop of os_proc_available_memory
//     since launch) below 3,000 MB with the entitlement").
//   Output 2: permitsPlanningEvent() returns false at serious or critical and counts the
//     skipped event (work item WI-32: "at serious or critical the planner's event is
//     skipped"; Apple: "At higher thermal states your app should reduce usage of system
//     resources").
// Both readings are taken again at every model step (ModelStepObserving) and at every due
// planning event; Apple's page says to "Use the returned value as advisory information
// only and don't cache it".
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave:
// - https://developer.apple.com/documentation/os/os_proc_available_memory.md (fetched
//   November 4, 2026): "extern size_t os_proc_available_memory();"; availability "iOS:
//   13.0.0 -", "macCatalyst: 13.1.0 -", "macOS: -"; module "os"; the return value quoted
//   above. size_t is imported into Swift as Int.
// - docs/apple_port/evidence/research1.json: ThermalState "has nominal, fair, serious and
//   critical cases" and "At higher thermal states your app should reduce usage of system
//   resources." (developer.apple.com/documentation/foundation/processinfo/
//   thermalstate-swift.property); "Memory limits can change during the app life cycle".
// - docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html REQ-B-14 ("memory readout
//   (os_proc_available_memory)"), REQ-B-15 (the peak footprint line), REQ-B-16
//   ("ProcessInfo.thermalState (nominal, fair, serious, critical) at start and end"),
//   section 5 ("MemoryMonitor ... ProcInfoMemoryMonitor").

import Combine
import Foundation
import os
import UEMACore

/// The two device readings the memory screen and the planning rule need.
public protocol MemoryStatusReading: AnyObject {
    /// Bytes the app may allocate before its memory limit (os_proc_available_memory); nil
    /// where the function does not exist (macOS).
    var availableMemoryBytes: Int? { get }

    /// The device's thermal state now.
    var thermalState: ProcessInfo.ThermalState { get }
}

/// MemoryStatusReading over os_proc_available_memory (iOS only) and
/// ProcessInfo.thermalState (specification section 5, "ProcInfoMemoryMonitor").
public final class ProcInfoMemoryMonitor: MemoryStatusReading {
    private let processInfo: ProcessInfo

    /// A monitor reading `processInfo` (the process's own by default).
    public init(processInfo: ProcessInfo = ProcessInfo.processInfo) {
        self.processInfo = processInfo
    }

    /// os_proc_available_memory() on iOS; nil on macOS, where the function is unavailable.
    public var availableMemoryBytes: Int? {
        #if os(iOS)
        return Int(os_proc_available_memory())
        #else
        return nil
        #endif
    }

    /// ProcessInfo.thermalState.
    public var thermalState: ProcessInfo.ThermalState {
        processInfo.thermalState
    }
}

/// Publishes the available memory in MB, the peak footprint since launch in MB and the
/// thermal state, and permits a due planning event only at thermal state nominal or fair.
@MainActor
public final class MemoryViewModel: ObservableObject, PlanningPermitting, ModelStepObserving {
    /// Bytes per MB: 1,048,576.
    public static let bytesPerMB = 1_048_576.0

    /// Available memory now, MB; nil where it cannot be read (macOS).
    @Published public private(set) var availableMemoryMB: Double? = nil

    /// Available memory at launch (this view model's creation), MB; nil on macOS.
    @Published public private(set) var launchAvailableMemoryMB: Double? = nil

    /// The largest drop of the available memory since launch, MB (REQ-B-15's peak
    /// footprint); 0 before any drop; nil on macOS.
    @Published public private(set) var peakFootprintMB: Double? = nil

    /// The thermal state at the last reading.
    @Published public private(set) var thermalState: ProcessInfo.ThermalState

    /// Due planning events this view model refused.
    @Published public private(set) var planningEventsSkipped = 0

    private let monitor: any MemoryStatusReading
    private let launchAvailableBytes: Int?

    /// A view model over `monitor`; the first reading is the launch reading.
    public init(monitor: any MemoryStatusReading) {
        self.monitor = monitor
        let launchBytes = monitor.availableMemoryBytes
        self.launchAvailableBytes = launchBytes
        self.thermalState = monitor.thermalState
        if let launchBytes = launchBytes {
            let launchMB = Double(launchBytes) / MemoryViewModel.bytesPerMB
            self.availableMemoryMB = launchMB
            self.launchAvailableMemoryMB = launchMB
            self.peakFootprintMB = 0
        }
    }

    /// True at serious and critical: the thermal states at which the planner's event is
    /// skipped.
    public nonisolated static func blocksPlanning(_ state: ProcessInfo.ThermalState) -> Bool {
        state == .serious || state == .critical
    }

    /// The thermal state's name as Apple spells the case.
    public nonisolated static func name(of state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal:
            return "nominal"
        case .fair:
            return "fair"
        case .serious:
            return "serious"
        case .critical:
            return "critical"
        @unknown default:
            return "unknown"
        }
    }

    /// The current thermal state's name.
    public var thermalStateName: String {
        MemoryViewModel.name(of: thermalState)
    }

    /// Takes both readings again.
    public func refresh() {
        thermalState = monitor.thermalState
        guard let bytes = monitor.availableMemoryBytes else {
            availableMemoryMB = nil
            return
        }
        availableMemoryMB = Double(bytes) / MemoryViewModel.bytesPerMB
        if let launchBytes = launchAvailableBytes {
            let dropMB = Double(launchBytes - bytes) / MemoryViewModel.bytesPerMB
            peakFootprintMB = max(peakFootprintMB ?? 0, dropMB)
        }
    }

    /// Reads the thermal state and returns false at serious or critical, counting the
    /// skipped event.
    public func permitsPlanningEvent() -> Bool {
        refresh()
        if MemoryViewModel.blocksPlanning(thermalState) {
            planningEventsSkipped += 1
            return false
        }
        return true
    }

    /// Takes both readings after every model step.
    public func observe(_ report: ModelStepReport) {
        refresh()
    }

    /// Clears the skipped count; the launch reading and the peak footprint stay, since
    /// they belong to the process, not to the session.
    public func reset() {
        planningEventsSkipped = 0
        refresh()
    }
}
