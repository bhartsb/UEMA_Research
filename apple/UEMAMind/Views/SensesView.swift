// SensesView.swift: the sense panel of UEMA Mind: per active sense its encoder input's
// shape, minimum and maximum, its token (192 numbers) as a 12 x 16 grid of cells, and the
// prediction error. Work item WI-38, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html (row WI-38: "SensesView: six
// tokens as 192-cell strips"); specification REQ-B-14: "Sense panel: per sense the
// input's shape, minimum and maximum, the token as a 12 x 16 heat map, the prediction
// error (mean squared difference)". The plan's "192-cell strip" and the specification's
// "12 x 16 heat map" are the same 192 cells; this file draws them as 12 rows of 16 cells,
// row r holding token numbers 16 r to 16 r + 15, as SensePanelRow.heatMap lays them
// (apple/UEMAMind/ViewModels/SensesViewModel.swift:74-81, WI-32).
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). SwiftUI only; no UIKit or AppKit type.
//
// INPUT: the rows SensesViewModel publishes after each model step (SensesViewModel.swift:
// 36-82, 84-130): per active sense the input's primary and auxiliary shapes, the primary
// minimum and maximum, whether the reading was fresh, the token, the 12 x 16 heat map and
// the prediction error (the mean over the 192 positions of (predicted - actual)^2 between
// the token predicted at the previous model step and this model step's token; nil on the
// first model step). Its anchors: 0 = the prediction equals the token; 1.0 = the error of
// predicting all zeros for a real encoder token, whose root mean square is 1.0
// (encoder_registry.py:53, quoted at apple/UEMAKit/Sources/UEMACore/SenseName.swift:30-31
// and :102). OPERATION: `TokenHeatScale` maps a token number to a colour: blue at -3 and
// below, white at 0, red at +3 and above, linear between; 3 is three times that root mean
// square of 1.0. A sense of `SenseName.active` with no row yet (before the first model
// step) is drawn with grey cells. OUTPUT: pixels; nothing is written back. intermind is
// inactive in the released run (SenseName.swift:76-90) and gets no block.
//
// The "First build on a Mac" checklist of this work item is in UEMAMindApp.swift's header.

import Foundation
import SwiftUI
import UEMACore

/// The colour of one token number in the heat map. One responsibility: the value-to-colour
/// rule, kept apart from the drawing so a test reads it.
struct TokenHeatScale {
    /// The value drawn at full colour: 3.0, three times the root mean square (1.0) of a
    /// real encoder token (encoder_registry.py:53). Larger magnitudes are clipped to it.
    static let fullScale: Float = 3

    /// The red, green and blue components (0 to 1) of `value`: (1, 1, 1) at 0; toward
    /// (1, 0, 0) for positive values, (0, 0, 1) for negative values, reached at
    /// `fullScale`; a non-finite value is drawn black (0, 0, 0).
    static func components(of value: Float) -> (red: Double, green: Double, blue: Double) {
        guard value.isFinite else { return (0, 0, 0) }
        let fraction = Double(min(abs(value) / fullScale, 1))
        if value >= 0 {
            return (1, 1 - fraction, 1 - fraction)
        }
        return (1 - fraction, 1 - fraction, 1)
    }

    /// The SwiftUI colour of `value`.
    static func color(of value: Float) -> Color {
        let c = components(of: value)
        return Color(red: c.red, green: c.green, blue: c.blue)
    }
}

/// The panel's wording. One responsibility: numbers and shapes into text.
enum SensePanelText {
    /// "[26, 19]" for a shape; "none" for nil.
    static func shapeText(_ shape: [Int]?) -> String {
        guard let shape else { return "none" }
        return "[" + shape.map(String.init).joined(separator: ", ") + "]"
    }

    /// "input shape [26, 19], second array none" with ", not refreshed this model step"
    /// appended for a reading that was not fresh.
    static func shapeLine(_ row: SensePanelRow) -> String {
        let fresh = row.isFresh ? "" : ", not refreshed this model step"
        return "input shape \(shapeText(row.inputShape)), second array \(shapeText(row.auxiliaryShape))\(fresh)"
    }

    /// A number with four significant digits, or "none" for nil.
    static func numberText(_ value: Float?) -> String {
        guard let value else { return "none" }
        return String(format: "%.4g", Double(value))
    }

    /// The six active senses in `SenseName.active` sequence, each with its row when one
    /// was given.
    static func blocks(_ rows: [SensePanelRow]) -> [SenseBlock] {
        SenseName.active.map { sense in SenseBlock(sense: sense, row: rows.first { $0.sense == sense }) }
    }
}

/// One block of the panel: an active sense and its row, nil before the first model step.
struct SenseBlock: Identifiable {
    /// The sense.
    let sense: SenseName

    /// Its row; nil when none was given.
    let row: SensePanelRow?

    var id: SenseName { sense }
}

/// The 192 numbers of one token drawn as 12 rows of 16 cells.
struct TokenHeatMap: View {
    /// 12 rows of 16 numbers (SensePanelRow.heatMap); nil draws grey cells.
    let heatMap: [[Float]]?

    var body: some View {
        Canvas { context, size in
            let rows = SensePanelRow.heatMapRows
            let columns = SensePanelRow.heatMapColumns
            let cellWidth = size.width / CGFloat(columns)
            let cellHeight = size.height / CGFloat(rows)
            for row in 0 ..< rows {
                for column in 0 ..< columns {
                    let rect = CGRect(
                        x: CGFloat(column) * cellWidth,
                        y: CGFloat(row) * cellHeight,
                        width: cellWidth,
                        height: cellHeight)
                    let color: Color
                    if let heatMap, row < heatMap.count, column < heatMap[row].count {
                        color = TokenHeatScale.color(of: heatMap[row][column])
                    } else {
                        color = Color.gray.opacity(0.25)
                    }
                    context.fill(Path(rect), with: .color(color))
                }
            }
        }
        .aspectRatio(
            CGFloat(SensePanelRow.heatMapColumns) / CGFloat(SensePanelRow.heatMapRows),
            contentMode: .fit)
        .accessibilityLabel("Token of 192 numbers as 12 rows of 16 cells")
    }
}

/// The six active senses, one block each.
struct SensesView: View {
    /// SensesViewModel.rows of the latest model step; empty before the first.
    let rows: [SensePanelRow]

    /// The model step the rows belong to; nil before the first.
    let step: ModelStep?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(heading)
                .font(.headline)
            Text("Cell colour: blue at -3, white at 0, red at +3 (3 = three times the root mean square of a real encoder token).")
                .font(.caption)
                .foregroundStyle(.secondary)
            ForEach(SensePanelText.blocks(rows)) { block in
                HStack(alignment: .top, spacing: 12) {
                    TokenHeatMap(heatMap: block.row?.heatMap)
                        .frame(width: 160)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(block.sense.rawValue).font(.subheadline.bold())
                        if let row = block.row {
                            Text(SensePanelText.shapeLine(row))
                            Text("input minimum \(SensePanelText.numberText(row.inputMinimum)), maximum \(SensePanelText.numberText(row.inputMaximum))")
                            Text("prediction error \(SensePanelText.numberText(row.predictionError)) (0 = predicted exactly; 1.0 = predicting all zeros)")
                        } else {
                            Text("no token yet")
                        }
                    }
                    .font(.caption.monospacedDigit())
                }
            }
        }
    }

    /// The panel heading: "Senses at model step 3: one token of 192 numbers per sense", or
    /// "Senses: no model step yet" before the first.
    private var heading: String {
        guard let step else { return "Senses: no model step yet" }
        return "Senses at model step \(step.count): one token of \(Token.count) numbers per sense"
    }
}

#if DEBUG
/// Synthetic rows for the preview: sense k's number i is sqrt(2) sin(i (k + 1) / 10), a
/// pattern of root mean square near 1.0. Not an encoder's output.
enum SensesPreviewData {
    static func rows() -> [SensePanelRow] {
        var out: [SensePanelRow] = []
        for (k, sense) in SenseName.active.enumerated() {
            let values: [Float] = (0 ..< Token.count).map { i in
                Float(2.0.squareRoot() * sin(Double(i * (k + 1)) / 10))
            }
            guard let token = try? Token(values: values) else { continue }
            out.append(SensePanelRow(
                sense: sense,
                inputShape: [4],
                auxiliaryShape: nil,
                inputMinimum: -1,
                inputMaximum: 1,
                isFresh: true,
                token: token,
                heatMap: SensePanelRow.heatMap(of: token),
                predictionError: 0.01 * Float(k + 1)))
        }
        return out
    }
}

#Preview("Senses, synthetic tokens") {
    ScrollView {
        SensesView(rows: SensesPreviewData.rows(), step: ModelStep(count: 3))
            .padding()
    }
}
#endif
