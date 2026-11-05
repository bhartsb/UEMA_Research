// TactileTokenizer.swift: the tactile tokenizer of the UEMA Apple port (work item WI-30, target B
// of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-5, the last
// stage). Nothing in this file was compiled on the machine that wrote it. The "First build on a
// Mac" checklist of this module is in VisionPatcher.swift's header.
//
// INPUT. The touch pooler's patch fields for one model step (the last env step's; touch_adapter.py
// `last_fields`, contract apple/UEMAKit/Specs/touch_pooler.md), float32 [n, 10] with n <= 256, one
// row per skin patch (one patch = one skin body, 53 in the released world): 0 normal pressure Pa;
// 1 shear Pa; 2 slip rate mm/s; 3 contact area mm^2; 4-7 vibration power N^2 at 5, 10, 15,
// 20 Hz; 8-9 the two principal stretch ratios (1.0 at rest). Optionally the side columns of
// each patch row, float32 [n, 10] (the layout `TactileTokenizer.sideColumnLayout` names):
// 0-2 the patch body's position relative to the root body in metres (`touch_pos`,
// env.py:3554); 3-4 the self-touch share and own-part count already scaled (`touch_self`);
// 5-9 the strongest contact's normal (3) in the body frame, its depth and its arrival speed,
// already scaled (`touch_contact`) - the pooler's columns of UEMASenses/Tactile/TouchPooler.swift.
// The tactile reader of work item WI-28 (UEMASenses/Tactile/TactileSense.swift, its
// `TactileReadingLayout`) gives a second layout, also accepted: `primary` [256, 10] with the
// patch rows first and the unused slots zero, and `auxiliary` [256, 11] whose columns 0-9 are the
// side columns above and whose column 10 is 1 for a patch row and 0 for an unused slot. Only the
// flagged rows are patch rows: touch.py:114-115 subtracts `_STRETCH_REST` from the n_patches rows
// only, and a zero slot re-zeroed as a patch would read -2 in both stretch columns and enter the
// point cloud as a contact. Without `auxiliary` every row of `primary` is a patch row
// (`tokenize_touch` with n_patches = the row count), so a reading padded with zero slots must
// carry the auxiliary array. A reader that has built the [32, 19] token itself may pass it as
// `primary` instead; it is checked and passed on.
//
// OPERATION.
//  1. `tokenize_touch` (Source/Python/tokenizer/modalities/touch.py:75-76, :114-117): columns 8
//     and 9 minus 1.0 (`_STRETCH_REST`), then every column divided by `_FIELD_NORMS` = [100,000
//     Pa, 50,000 Pa, 200 mm/s, 20,000 mm^2, 10, 10, 10, 10 N^2, 0.5, 0.5], in float32; rows n to
//     255 are zero.
//  2. `touch_point_cloud` (Source/Python/mimo_bridge/lem_model.py:224-264): the active rows (any
//     nonzero number among the 10) among the first n, sorted by normalised pressure (column 0)
//     descending (equal pressures keep slot sequence; torch.argsort fixes none), the first 32
//     (`TOUCH_TOKENS`, :96) kept. Token columns: 0-2 position / 0.5 m (`TOUCH_POS_SCALE`, :99;
//     zero without side columns), 3-10 the first 8 normalised fields, 11-12 the self columns,
//     13-17 the contact columns, 18 = 1.0 (the valid flag, `TOUCH_VALID_COL`). Unused rows are 0.
//
// OUTPUT. float32 [32, 19] (`TOUCH_TOKENS` x `TOUCH_TOKEN_F`, lem_model.py:96-97), the tactile
// encoder's input.

import Foundation
import UEMACore

/// The tactile tokenizer: patch fields over `_FIELD_NORMS`, then the 32 strongest patches as the
/// tactile encoder's [32, 19] input (touch.py:75-76, :114-117; lem_model.py:224-264).
public struct TactileTokenizer: Tokenizer, Sendable {
    /// `_FIELD_NORMS` (touch.py:75): pressure 100,000 Pa, shear 50,000 Pa, slip 200 mm/s, area
    /// 20,000 mm^2, four vibration bands 10 N^2 each, two stretches 0.5 each.
    public static let fieldNorms: [Float] = [100_000, 50_000, 200, 20_000, 10, 10, 10, 10, 0.5, 0.5]

    /// `_STRETCH_REST` (touch.py:76): the raw stretch ratio at rest, subtracted from columns 8, 9.
    public static let stretchRest: Float = 1.0

    /// Fields per patch: 10.
    public static let fieldCount = 10

    /// Patch slots of the frame: 256 (`TOUCH_MAX_IMPULSES`, tokenizer/config.py:51).
    public static let slotCount = 256

    /// Token rows: 32 (`TOUCH_TOKENS`, lem_model.py:96).
    public static let tokenRows = 32

    /// Token columns: 19 (`TOUCH_TOKEN_F`, lem_model.py:97).
    public static let tokenColumns = 19

    /// The valid-flag column: 18 (`TOUCH_VALID_COL` = `TOUCH_TOKEN_F` - 1).
    public static let validColumn = 18

    /// Metres per unit of the position columns: 0.5 (`TOUCH_POS_SCALE`, lem_model.py:99).
    public static let positionScale: Float = 0.5

    /// The side-column layout of `auxiliary`, by column: position x, y, z (m, root-relative),
    /// self share, self parts, contact normal x, y, z, depth, arrival speed.
    public static let sideColumnLayout = [
        "position_x_m", "position_y_m", "position_z_m", "self_share", "self_parts",
        "contact_normal_x", "contact_normal_y", "contact_normal_z", "contact_depth", "contact_arrival",
    ]

    /// Columns of the side-column layout: 10.
    public static let sideColumnCount = 10

    /// Columns of the WI-28 reader's auxiliary array: 10 side columns plus the patch-row flag
    /// (`TactileReadingLayout.auxiliaryColumnCount` 11, UEMASenses/Tactile/TactileSense.swift).
    public static let readerAuxiliaryColumnCount = 11

    /// The patch-row flag column of the WI-28 reader's auxiliary array: 10
    /// (`TactileReadingLayout.patchRowColumn`, UEMASenses/Tactile/TactileSense.swift).
    public static let patchRowFlagColumn = 10

    /// The sense served: tactile.
    public var sense: SenseName { .tactile }

    /// A tokenizer; it holds no state.
    public init() {}

    /// Step 1 of this file's header with every row of `fields` a patch row. INPUT: `fields`
    /// [n][10], n <= 256. OUTPUT: [256][10].
    public static func normalisedSlots(_ fields: ShapedArray) throws -> [[Float]] {
        let n = try fieldRowCount(fields)
        return normalisedSlots(patchRows: (0 ..< n).map { row(of: fields, $0, width: fieldCount) })
    }

    /// Step 1 of this file's header. INPUT: the raw fields of the patch rows, at most 256 rows of
    /// 10 numbers (a longer list is cut at 256; a shorter row leaves its missing columns zero).
    /// OUTPUT: [256][10]: patch row k in slot k, columns 8 and 9 minus 1.0, every column over
    /// `_FIELD_NORMS`, float32; slots past the patch rows zero (touch.py:109-117).
    public static func normalisedSlots(patchRows: [[Float]]) -> [[Float]] {
        var slots = [[Float]](repeating: [Float](repeating: 0, count: fieldCount), count: slotCount)
        for (r, fields) in patchRows.prefix(slotCount).enumerated() {
            for c in 0 ..< min(fieldCount, fields.count) {
                let value = c >= 8 ? fields[c] - stretchRest : fields[c]
                slots[r][c] = value / fieldNorms[c]
            }
        }
        return slots
    }

    /// The row count of a [n <= 256, 10] field array; throws `TokenizerError.wrongShape` otherwise.
    static func fieldRowCount(_ fields: ShapedArray) throws -> Int {
        let expected = "[n <= 256, 10] or [32, 19]"
        let n = try TokenizerChecks.rows(
            of: fields, columns: [fieldCount], sense: .tactile, name: "primary", expected: expected)
        guard n <= slotCount else {
            throw TokenizerError.wrongShape(sense: .tactile, array: "primary", expected: expected, found: fields.shape)
        }
        return n
    }

    /// The first `width` numbers of row `index` of a rank-2 array whose rows hold at least
    /// `width` numbers.
    static func row(of array: ShapedArray, _ index: Int, width: Int) -> [Float] {
        let start = index * array.shape[1]
        return Array(array.values[start ..< start + width])
    }

    /// Step 2 of this file's header. INPUT: `slots` [256][10] from `normalisedSlots`; `sides`
    /// [n][10] in `sideColumnLayout` for the n patch rows, or nil (then every slot is read and
    /// the position columns stay zero). OUTPUT: [32 x 19] row-major.
    public static func pointCloud(slots: [[Float]], sides: [[Float]]?) -> [Float] {
        var out = [Float](repeating: 0, count: tokenRows * tokenColumns)
        let n = sides.map { min(slots.count, $0.count) } ?? slots.count
        var active: [Int] = []
        for i in 0 ..< n where slots[i].contains(where: { $0 != 0 }) {
            active.append(i)
        }
        let ordered = active.sorted { a, b in
            if slots[a][0] != slots[b][0] { return slots[a][0] > slots[b][0] }
            return a < b
        }
        for (k, i) in ordered.prefix(tokenRows).enumerated() {
            let o = k * tokenColumns
            // A side row shorter than 10 numbers fills the columns it has (lem_model.py:243-262
            // reads each column group only when it is present).
            if let side = sides?[i] {
                for j in 0 ..< min(3, side.count) {
                    out[o + j] = side[j] / positionScale
                }
                for j in 0 ..< max(0, min(7, side.count - 3)) {
                    out[o + 11 + j] = side[3 + j]
                }
            }
            for j in 0 ..< min(8, slots[i].count) {
                out[o + 3 + j] = slots[i][j]
            }
            out[o + validColumn] = 1.0
        }
        return out
    }

    /// The encoder input [32, 19] of one reading (any layout of this file's header);
    /// `isFresh` carried.
    public func tokenize(_ reading: SenseReading) throws -> EncoderInput {
        try TokenizerChecks.requireSense(reading, .tactile)
        let primary = reading.primary
        if primary.shape == [TactileTokenizer.tokenRows, TactileTokenizer.tokenColumns] {
            return EncoderInput(sense: .tactile, primary: primary, isFresh: reading.isFresh)
        }
        let rowCount = try TactileTokenizer.fieldRowCount(primary)
        var patchRows = Array(0 ..< rowCount)
        var sides: [[Float]]?
        if let auxiliary = reading.auxiliary {
            let expected = "[\(rowCount), 10] side columns or [\(rowCount), 11] with the patch-row flag"
            let auxiliaryRows = try TokenizerChecks.rows(
                of: auxiliary,
                columns: [TactileTokenizer.sideColumnCount, TactileTokenizer.readerAuxiliaryColumnCount],
                sense: .tactile, name: "auxiliary", expected: expected)
            guard auxiliaryRows == rowCount else {
                throw TokenizerError.wrongShape(
                    sense: .tactile, array: "auxiliary", expected: expected, found: auxiliary.shape)
            }
            let width = auxiliary.shape[1]
            if width == TactileTokenizer.readerAuxiliaryColumnCount {
                patchRows = patchRows.filter {
                    auxiliary.values[$0 * width + TactileTokenizer.patchRowFlagColumn] > 0.5
                }
            }
            sides = patchRows.map { TactileTokenizer.row(of: auxiliary, $0, width: TactileTokenizer.sideColumnCount) }
        }
        let fields = patchRows.map { TactileTokenizer.row(of: primary, $0, width: TactileTokenizer.fieldCount) }
        let slots = TactileTokenizer.normalisedSlots(patchRows: fields)
        let token = TactileTokenizer.pointCloud(slots: slots, sides: sides)
        return EncoderInput(
            sense: .tactile,
            primary: try ShapedArray(shape: [TactileTokenizer.tokenRows, TactileTokenizer.tokenColumns], values: token),
            isFresh: reading.isFresh)
    }
}
