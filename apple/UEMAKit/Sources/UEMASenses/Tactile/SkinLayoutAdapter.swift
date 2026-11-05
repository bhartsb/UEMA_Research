// SkinLayoutAdapter.swift: work item WI-26's decoded skin layout (UEMAScene, SkinLayout.swift)
// presented as this module's `SkinTable` (work item WI-28, target B; specification section 5:
// the `SkinTable` stage, concrete type `ShippedSkinTable` in SkinSpread.swift).
//
// WHAT THIS FILE IS. One conversion: `SkinLayout` (its `bodies`, each with `submeshes`)
// into `SkinBodyRows` and `SkinMeshRows`, the row types `LinearContactSpreader` searches.
// The numbers are copied, not recomputed: body id, name, pooler row, the two scales in
// metres, the point offset, and per submesh the vertex positions (SIMD3<Double> to Vec3),
// the active mask, the sensor indices and the neighbour lists. The live positions are
// rebuilt from the masks in output order and checked against the sensor indices
// (`SkinBodyRows.init?`), so a layout whose indices do not run 0, 1, 2 ... is refused.
//
// WHY TWO TABLE TYPES. `ShippedSkinTable` decodes the WI-13 JSON on its own, so this
// module's tests run against the file without the UEMAScene types; a session that already
// holds a `SkinLayout` (loaded through `SkinLayout.loadShipped`, which also verifies the
// digest) wraps it here instead of decoding the 849,895-byte file twice.
//
// API READ (SkinLayout.swift of UEMAScene at 5:44 PM PST, November 4, 2026): `SkinLayout`
// with `bodies: [SkinBody]`, `pointCount`, `layoutDigest`, `body(id:)`,
// `static func loadShipped(from:)`; `SkinBody` with `bodyID`, `bodyName`, `poolerRow`,
// `scaleM`, `forceSpreadScaleM`, `sensorCount`, `pointOffset`, `submeshes`; `SkinSubmesh`
// with `submesh`, `geomID`, `geomName`, `geomType`, `geomSizeM: SIMD3<Double>`,
// `isPointCloud`, `positionsM: [SIMD3<Double>]`, `active`, `sensorIndex`, `neighbours`.
// A change to those names breaks only this file.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain); the "First build on a Mac" checklist is in TouchPooler.swift's header.

import Foundation
import UEMAScene

/// `SkinLayout` wrapped as a `SkinTable`.
public struct SkinLayoutTable: SkinTable {
    /// The wrapped layout.
    public let layout: SkinLayout
    /// The converted rows, body id order.
    public let bodies: [SkinBodyRows]
    private let index: [Int: Int]

    /// Converts every body; nil when a body's sensor indices do not run 0, 1, 2 ... over
    /// its active vertices in submesh then vertex order.
    public init?(layout: SkinLayout) {
        self.layout = layout
        var rows: [SkinBodyRows] = []
        var map: [Int: Int] = [:]
        for body in layout.bodies {
            let meshes = body.submeshes.map { sub in
                SkinMeshRows(
                    submesh: sub.submesh, geomId: sub.geomID, geomName: sub.geomName,
                    geomType: sub.geomType,
                    geomSize: [sub.geomSizeM.x, sub.geomSizeM.y, sub.geomSizeM.z],
                    isPointCloud: sub.isPointCloud,
                    positions: sub.positionsM.map { Vec3($0.x, $0.y, $0.z) },
                    active: sub.active, sensorIndex: sub.sensorIndex, neighbours: sub.neighbours)
            }
            guard let row = SkinBodyRows(
                bodyId: body.bodyID, bodyName: body.bodyName, scale: body.scaleM,
                forceSpreadScale: body.forceSpreadScaleM, poolerRow: body.poolerRow,
                pointOffset: body.pointOffset, submeshes: meshes) else { return nil }
            map[row.bodyId] = rows.count
            rows.append(row)
        }
        bodies = rows
        index = map
    }

    /// The shipped layout of the UEMAScene resource bundle, converted. Throws what
    /// `SkinLayout.loadShipped` throws, or `SkinTableError.countMismatch` when the
    /// conversion refuses a body.
    public static func shipped() throws -> SkinLayoutTable {
        let layout = try SkinLayout.loadShipped()
        guard let table = SkinLayoutTable(layout: layout) else {
            throw SkinTableError.countMismatch("a body's sensor_index does not run over its active vertices")
        }
        return table
    }

    /// The body with this id, or nil.
    public func body(id: Int) -> SkinBodyRows? {
        guard let i = index[id] else { return nil }
        return bodies[i]
    }

    /// `layout.pointCount`.
    public var pointCount: Int { layout.pointCount }

    /// `layout.layoutDigest`.
    public var layoutDigest: String { layout.layoutDigest }
}
