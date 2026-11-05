// CommandView.swift: the command (26 body handles x 9 numbers) as a table. Work item
// WI-38, target B of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html (row WI-38:
// "CommandView: the 26 x 9 table"); specification REQ-B-14: "the command as a 26 x 9
// table".
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). SwiftUI only (`Grid`, macOS 13 and iOS 16 or newer,
// inside the project's macOS 14 and iOS 17 floors); no UIKit or AppKit type.
//
// INPUT: the rows CommandViewModel publishes after each model step
// (apple/UEMAMind/ViewModels/CommandViewModel.swift:29-42, 44-103, WI-32): one
// `CommandTableRow` per body handle in `BodyHandle` sequence (the 24 of HANDLE_NAMES then
// lower_trunk and chest, apple/UEMAKit/Sources/UEMACore/Command.swift:86-166), each with
// the handle's Python name and its 9 numbers (target position x y z, orientation
// quaternion x y z w, radius, strength; Command.swift:168-191), and the position unit:
// "cm" for the sent command (the held, executed command the decoder read) or "cm / 200"
// for the raw command (CommandViewModel.positionUnit). OPERATION: `CommandTable.cells`
// formats each number with 3 decimals. OUTPUT: pixels; nothing is written back.
//
// The "First build on a Mac" checklist of this work item is in UEMAMindApp.swift's header.

import Foundation
import SwiftUI
import UEMACore

/// The command table's number formatting. One responsibility: rows of numbers into text
/// cells, kept apart from the view so a test reads it.
enum CommandTable {
    /// The 9 column headers in `CommandColumn` sequence, the same words as
    /// CommandViewModel.columnNames (CommandViewModel.swift:49); kept here as a constant of
    /// this non-isolated type so a view reads it outside the main actor's isolation.
    static let columnHeaders = ["x", "y", "z", "qx", "qy", "qz", "qw", "radius", "strength"]

    /// 26 rows of 9 cells, each number with 3 decimals.
    static func cells(_ rows: [CommandTableRow]) -> [[String]] {
        rows.map { row in row.values.map { String(format: "%.3f", Double($0)) } }
    }
}

/// The command as 26 rows of 9 numbers with the handle name on each row.
struct CommandView: View {
    /// CommandViewModel's rows; empty before the first model step.
    let rows: [CommandTableRow]

    /// The unit of the three position columns: "cm" or "cm / 200".
    let positionUnit: String

    /// What the table shows, such as "sent to the decoder at model step 12, latched at
    /// this model step".
    let caption: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Command: 26 body handles x 9 numbers").font(.headline)
            if rows.isEmpty {
                Text("No command yet: the first one is made on the first model step.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("\(caption); position in \(positionUnit); quaternion x y z w; radius 0 to 1; strength 0 to 1 after the decoder's clip")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                table
            }
        }
    }

    private var table: some View {
        let cells = CommandTable.cells(rows)
        return ScrollView([.horizontal, .vertical]) {
            Grid(alignment: .trailing, horizontalSpacing: 10, verticalSpacing: 2) {
                GridRow {
                    Text("handle").gridColumnAlignment(.leading)
                    ForEach(CommandTable.columnHeaders, id: \.self) { name in
                        Text(name)
                    }
                }
                .font(.caption.bold())
                ForEach(0 ..< rows.count, id: \.self) { index in
                    GridRow {
                        Text(rows[index].name)
                        ForEach(0 ..< cells[index].count, id: \.self) { column in
                            Text(cells[index][column])
                        }
                    }
                    .font(.caption.monospacedDigit())
                }
            }
        }
    }
}

#Preview("Command, all zeros except gaze straight ahead") {
    var command = Command.zero(scale: .centimetres)
    command.setGaze(.straightAhead)
    return CommandView(
        rows: CommandViewModel.rows(of: command),
        positionUnit: CommandViewModel.positionUnit(of: .centimetres),
        caption: "preview")
        .padding()
}
