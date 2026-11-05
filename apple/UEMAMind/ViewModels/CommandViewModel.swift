// CommandViewModel.swift: the command as a table of 26 body handles x 9 numbers for the
// command screen. Work item WI-32, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html. The work item's main file and
// its "First build on a Mac" checklist are RunViewModel.swift.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1).
//
// WHAT IT PUBLISHES after every model step: the command the decoder read (the held,
// executed command, positions in cm, ModelStepReport.sentCommand) as 26 rows, one per
// handle in the command's row sequence with the Python handle name, each with its 9
// numbers (target position x y z in cm, orientation quaternion x y z w, radius, strength);
// the raw command of the same model step (ModelStepReport.rawCommand, positions in
// position_range units, cm / 200); whether the sent command was latched this model step;
// and the gaze pair (azimuth, elevation in radians) read from the eyes' row.
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave:
// - plan row WI-38: "CommandView: the 26 x 9 table."
// - apple/UEMAKit/Sources/UEMACore/Command.swift: BodyHandle (26 rows, pythonName),
//   CommandColumn (positionX 0 ... radius 7, strength 8), Command.row(_:), CommandScale,
//   Command.gazeAngles.
// - README.md lines 19-20 (per Command.swift): "A command is 9 numbers for each of 26 body
//   handles (target position, rotation, radius, strength)".

import Combine
import Foundation
import UEMACore

/// One handle's row of the command table.
public struct CommandTableRow: Identifiable, Equatable {
    /// The handle.
    public let handle: BodyHandle

    /// The handle, as the row's identity.
    public var id: BodyHandle { handle }

    /// The handle's name in the Python body layout (for example "wrist_L", "gaze").
    public var name: String { handle.pythonName }

    /// The 9 numbers in CommandColumn sequence.
    public let values: [Float]
}

/// Publishes the command of the last model step as a 26 x 9 table.
@MainActor
public final class CommandViewModel: ObservableObject, ModelStepObserving {
    /// The 9 column headings in CommandColumn sequence; the position unit depends on the
    /// table's scale (`positionUnit`).
    public static let columnNames = ["x", "y", "z", "qx", "qy", "qz", "qw", "radius", "strength"]

    /// The rows of the sent command (positions in cm); empty before the first model step.
    @Published public private(set) var sentRows: [CommandTableRow] = []

    /// The rows of the raw command (positions in cm / 200); empty before the first model
    /// step.
    @Published public private(set) var rawRows: [CommandTableRow] = []

    /// True when the sent command was latched at the last model step (once every 5 model
    /// steps under the released hold).
    @Published public private(set) var latchedThisModelStep = false

    /// The gaze pair of the last executed command; nil before the first model step.
    @Published public private(set) var gaze: GazeAngles? = nil

    /// The model step of the table; nil before the first.
    @Published public private(set) var step: ModelStep? = nil

    /// An empty table.
    public init() {}

    /// The 26 rows of `command`, one per handle in row sequence.
    public nonisolated static func rows(of command: Command) -> [CommandTableRow] {
        BodyHandle.allCases.map { CommandTableRow(handle: $0, values: command.row($0)) }
    }

    /// The position unit of a command at `scale`: "cm" or "cm / 200".
    public nonisolated static func positionUnit(of scale: CommandScale) -> String {
        switch scale {
        case .centimetres:
            return "cm"
        case .normalized:
            return "cm / 200"
        }
    }

    /// Builds both tables from `report`.
    public func observe(_ report: ModelStepReport) {
        sentRows = CommandViewModel.rows(of: report.sentCommand)
        rawRows = CommandViewModel.rows(of: report.rawCommand)
        latchedThisModelStep = report.latchedNewCommand
        gaze = report.executedCommand.gazeAngles
        step = report.step
    }

    /// Empties the tables.
    public func reset() {
        sentRows = []
        rawRows = []
        latchedThisModelStep = false
        gaze = nil
        step = nil
    }
}
