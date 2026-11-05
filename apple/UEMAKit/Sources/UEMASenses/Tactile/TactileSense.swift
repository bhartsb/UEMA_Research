// TactileSense.swift: the tactile sense as UEMACore's `SenseReader` (work item WI-28,
// target B; specification section 5: `TactileSense` over the four REQ-B-5 stages; plan
// section 2, stage 2; REQ-B-3 "a tactile read every physics substep").
//
// WHAT THIS FILE IS. The session-facing reader. It holds a `TactileReader` (TouchPooler.swift:
// contact reader, spreader, pooler, point-cloud builder) and turns each `PhysicsReading` into
// a `PhysicsContactView` (PhysicsContactView.swift) before handing it on.
//
// CADENCE (Protocols.swift `SenseReader`). `observePhysicsStep` 10 times per model step:
// contacts read and spread over the skin points (env.py:2360-2361). `observeEnvStep` 5 times
// per model step: contacts read again with the current velocities and the pooler updated
// (env.py:3116-3120; dt 0.01 s, the 100-sample vibration window covers 1 s). `read` once per
// model step: the last env step's fields.
//
// OUTPUT: one `SenseReading` of sense `.tactile`, n = the pooler's patch count (53 for the
// shipped table), the layout UEMATokenizer's `TactileTokenizer` (WI-30,
// TactileTokenizer.swift:6-15, `sideColumnLayout`) reads:
//   primary   [n, 10] float32: the raw patch fields of the last env step (pressure Pa,
//             shear Pa, slip mm/s, area mm2, vibration N^2 x4, stretch ratio x2), one row
//             per patch, the `patches` [n, 10] that `write_touch` stores (frame_encoder.py:
//             116-130). The tokenizer's divisors are not applied (Protocols.swift
//             `SenseReading`: "before the tokenizer's divisors"). The frame's 256 slots are
//             not padded here: `tokenize_touch` subtracts `_STRETCH_REST` from the n patch
//             rows only and zero-fills the rest (tokenizer/modalities/touch.py:111-117), and
//             a zero padding row would read (0 - 1.0) / 0.5 = -2 in columns 8-9 after a
//             subtraction over all 256 rows, an active row.
//   auxiliary [n, 10] float32, `TactileReadingLayout`: columns 0-2 the patch body's
//             position minus the root body's, metres, world axes (env.py:3554 `tpos`);
//             3-4 the self-touch columns (share 0 to 1, distinct own parts / 4);
//             5-9 the push columns (normal in the body frame x3, depth cm / 0.1, arrival
//             speed cm/s / 2.0).
//   Before the first env step (and after `reset`) the reading holds n rest rows (columns 8
//   and 9 = 1.0, every other number 0, auxiliary 0) with `isFresh` false: the pooler's own
//   output with no contact, which the tokenizer turns into the all-zero token.
//   `PressureSortedPointCloud.build(reading:)` below turns this reading into the [32, 19]
//   token (lem_model.py:221-264) inside this module.
//
// SOLID. Single responsibility: cadence and array packing. Dependency inversion: the tactile
// reader, the model tables and the velocity rule are injected.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain); the "First build on a Mac" checklist is in TouchPooler.swift's header.

import Foundation
import UEMACore

/// The layout of the tactile `SenseReading`: one row per patch in both arrays.
public enum TactileReadingLayout {
    /// The most rows either array may hold: `TOUCH_MAX_IMPULSES` (tokenizer/config.py:51).
    public static let slotCount = 256
    /// Columns of the primary array (`N_PATCH_FIELDS`, touch_adapter.py:33).
    public static let fieldCount = 10
    /// Columns of the auxiliary array (TactileTokenizer.sideColumnLayout's 10 names).
    public static let auxiliaryColumnCount = 10
    /// First of the three position columns, metres relative to the root body.
    public static let positionColumn = 0
    /// First of the two self-touch columns.
    public static let selfColumn = 3
    /// First of the five push columns.
    public static let contactColumn = 5
    /// The raw stretch ratio of a patch at rest, columns 8 and 9 (touch_adapter.py:221-222).
    public static let restStretch: Float = 1.0
}

/// What can go wrong when the tactile reading is packed or unpacked.
public enum TactileReadingError: Error, Equatable {
    /// The reading is not of sense `.tactile`.
    case wrongSense(SenseName)
    /// An array has a shape other than the layout's.
    case wrongShape(name: String, expected: [Int], found: [Int])
    /// None of the named bodies exists in the model.
    case missingBody(String)
}

/// The tactile sense over the physics. Specification section 5: `TactileSense`.
public final class TactileSense: SenseReader {
    /// `.tactile`.
    public let sense: SenseName = .tactile
    /// The four stages.
    public let reader: TactileReader
    /// geom_bodyid, body_rootid and body names of the loaded model.
    public let tables: ContactModelTables
    /// The contact-point velocity rule.
    public let velocity: PointVelocityReading

    /// Makes the sense from its injected parts.
    public init(reader: TactileReader, tables: ContactModelTables, velocity: PointVelocityReading) {
        self.reader = reader
        self.tables = tables
        self.velocity = velocity
    }

    /// The released configuration (`TactileReader.released`) over the shipped table: the root
    /// body is the first of "robot:hip", "hip", "robot:torso", "torso" in the model
    /// (env.py:1223-1229); the infant's own root is `body_rootid` of "hip" (env.py:1200-1202).
    /// `reproducesFrameNegation` true (the default) reproduces MIMo's in-place normal
    /// negation as the trainer met it (ContactReader.swift header; `TactileReader`).
    /// Throws `TactileReadingError.missingBody` when no such body exists.
    public static func released(table: SkinTable, tables: ContactModelTables,
                                velocity: PointVelocityReading,
                                reproducesFrameNegation: Bool = true) throws -> TactileSense {
        let candidates = ["robot:hip", "hip", "robot:torso", "torso"]
        guard let root = candidates.lazy.compactMap({ tables.bodyId(named: $0) }).first else {
            throw TactileReadingError.missingBody(candidates.joined(separator: ", "))
        }
        let reader = try TactileReader.released(table: table, rootBodyId: root,
                                                ownRootBodyId: tables.infantRootBodyId,
                                                reproducesFrameNegation: reproducesFrameNegation)
        return TactileSense(reader: reader, tables: tables, velocity: velocity)
    }

    private func view(_ physics: any PhysicsReading) -> PhysicsContactView {
        PhysicsContactView(physics: physics, tables: tables, velocity: velocity)
    }

    /// Contacts read and spread (10 times per model step).
    public func observePhysicsStep(_ physics: any PhysicsReading) throws {
        reader.readPhysicsStep(view(physics))
    }

    /// Contacts read again and the pooler updated (5 times per model step).
    public func observeEnvStep(_ physics: any PhysicsReading) throws {
        reader.poolEnvStep(view(physics))
    }

    /// The last env step's fields and columns as the two arrays of the file header; the
    /// rest rows, with `isFresh` false, before the first env step.
    public func read(_ physics: any PhysicsReading, at step: ModelStep) throws -> SenseReading {
        let slots = TactileReadingLayout.slotCount
        let fieldCount = TactileReadingLayout.fieldCount
        let auxCount = TactileReadingLayout.auxiliaryColumnCount
        guard let fields = reader.lastFields else {
            let n = reader.pooler.patchCount
            if n > slots {
                throw TactileTokenError.tooManyPatches(count: n, slots: slots)
            }
            var rest = [Float](repeating: 0, count: n * fieldCount)
            for p in 0 ..< n {
                rest[p * fieldCount + 8] = TactileReadingLayout.restStretch
                rest[p * fieldCount + 9] = TactileReadingLayout.restStretch
            }
            let restArray = try ShapedArray(shape: [n, fieldCount], values: rest)
            return SenseReading(
                sense: .tactile,
                primary: restArray,
                auxiliary: ShapedArray(zerosOfShape: [n, auxCount]),
                isFresh: false)
        }
        let n = fields.patchCount
        if n > slots {
            throw TactileTokenError.tooManyPatches(count: n, slots: slots)
        }
        let positions = reader.patchPositions(view(physics))
        var primary = [Float](repeating: 0, count: n * fieldCount)
        var auxiliary = [Float](repeating: 0, count: n * auxCount)
        for p in 0 ..< n {
            let row = fields.fields[p]
            for c in 0 ..< min(fieldCount, row.count) {
                primary[p * fieldCount + c] = row[c]
            }
            let a = p * auxCount
            if p < positions.count {
                auxiliary[a + TactileReadingLayout.positionColumn] = Float(positions[p].x)
                auxiliary[a + TactileReadingLayout.positionColumn + 1] = Float(positions[p].y)
                auxiliary[a + TactileReadingLayout.positionColumn + 2] = Float(positions[p].z)
            }
            if p < fields.selfColumns.count {
                for j in 0 ..< min(2, fields.selfColumns[p].count) {
                    auxiliary[a + TactileReadingLayout.selfColumn + j] = fields.selfColumns[p][j]
                }
            }
            if p < fields.contactColumns.count {
                for j in 0 ..< min(5, fields.contactColumns[p].count) {
                    auxiliary[a + TactileReadingLayout.contactColumn + j] = fields.contactColumns[p][j]
                }
            }
        }
        let primaryArray = try ShapedArray(shape: [n, fieldCount], values: primary)
        let auxiliaryArray = try ShapedArray(shape: [n, auxCount], values: auxiliary)
        return SenseReading(sense: .tactile, primary: primaryArray, auxiliary: auxiliaryArray, isFresh: true)
    }

    /// Clears the pooler and the last reads.
    public func reset() {
        reader.reset()
    }
}

extension PressureSortedPointCloud {
    /// The [32, 19] token from a tactile `SenseReading` laid out as `TactileReadingLayout`:
    /// the n patch rows are normalised by `TactileFieldNorms.normalise`
    /// (tokenizer/modalities/touch.py:109-117), then `build` runs with the positions and the
    /// two column groups of the auxiliary array (lem_model.py:221-264). Throws on another
    /// sense, on a primary array other than [n <= 256, 10] or on an auxiliary array other
    /// than [n, 10].
    public func build(reading: SenseReading) throws -> TactileToken {
        guard reading.sense == .tactile else { throw TactileReadingError.wrongSense(reading.sense) }
        let slots = TactileReadingLayout.slotCount
        let fieldCount = TactileReadingLayout.fieldCount
        let auxCount = TactileReadingLayout.auxiliaryColumnCount
        let shape = reading.primary.shape
        guard shape.count == 2, shape[1] == fieldCount, shape[0] <= slots else {
            throw TactileReadingError.wrongShape(name: "primary", expected: [slots, fieldCount], found: shape)
        }
        let n = shape[0]
        guard let aux = reading.auxiliary, aux.shape == [n, auxCount] else {
            throw TactileReadingError.wrongShape(name: "auxiliary", expected: [n, auxCount],
                                                 found: reading.auxiliary?.shape ?? [])
        }
        var fields: [[Float]] = []
        var positions: [Vec3] = []
        var selfColumns: [[Float]] = []
        var contactColumns: [[Float]] = []
        for p in 0 ..< n {
            let f = p * fieldCount
            fields.append(Array(reading.primary.values[f ..< f + fieldCount]))
            let a = p * auxCount
            positions.append(Vec3(Double(aux.values[a]), Double(aux.values[a + 1]), Double(aux.values[a + 2])))
            let s = a + TactileReadingLayout.selfColumn
            selfColumns.append(Array(aux.values[s ..< s + 2]))
            let c = a + TactileReadingLayout.contactColumn
            contactColumns.append(Array(aux.values[c ..< c + 5]))
        }
        let normalised = try TactileFieldNorms.normalise(fields, slots: slots)
        return build(normalisedSlots: normalised, patchPositions: positions,
                     selfColumns: selfColumns, contactColumns: contactColumns)
    }
}
