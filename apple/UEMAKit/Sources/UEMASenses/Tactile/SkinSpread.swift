// SkinSpread.swift: the second tactile stage of the UEMA Apple port (work item WI-28,
// target B; specification REQ-B-5 and section 5: `SkinTable` and `ContactSpreader`;
// contract apple/UEMAKit/Specs/touch_pooler.md section 1, row `point_forces`).
//
// WHAT THIS FILE IS. The shipped skin table (`ShippedSkinTable`: the JSON that
// apple/scripts/export_skin_layout.py writes, work item WI-13, format "uema.skin_layout.v1")
// and the spread of one contact's force over that body's skin points
// (`LinearContactSpreader`): the port of `conservative_spread_linear`
// (Source/Python/mimo_bridge/contact_mechanics.py:60-76, installed as MIMo's response
// function at env.py:1172-1173) and of the search it calls,
// `TrimeshTouch.get_sensors_within_distance` and `_nearest_within_distance_search`
// (third_party/MIMo/mimo/mimo_infant/touch/touch.py:1297-1358 and :1238-1293).
//
// INPUT. A `SkinContact` (ContactReader.swift) with `positionBody` (metres, body frame) and
// `forceBody` (N, body frame); the body's rows of the skin table: every vertex of every
// submesh (one submesh per colliding geom, touch.py:884-889) with its body-frame position,
// its active flag (false for a vertex inside another geom of the same body, touch.py:917-925),
// its index among the body's live points and its neighbour list (the mesh's edges,
// touch.py:1222-1235); the body's `force_spread_scale_m` (= sensor spacing / touch_scale_mult
// 1.0, env.py:1170).
//
// OPERATION (contact_mechanics.py:67-76). scale = force_spread_scale_m; limit = 3 scale;
// search distance = limit x 1.5 (touch.py:1343 `distance_limit * distance_margin`, margin
// 1.5). Per submesh (touch.py:1256-1293): d_v = |vertex_v - pos| for every vertex (active or
// not, touch.py:1362-1378); skip the submesh when its nearest vertex is farther than the
// search distance; a one-vertex submesh contributes that vertex when active; otherwise a
// graph search from the nearest vertex and its neighbours that visits every vertex with
// d_v < search distance reachable through such vertices, and collects the active ones.
// Keep the collected vertices with d_v < limit (touch.py:1352-1356, strict). Weights
// w_v = max(1 - d_v / (2 scale), 0); when no vertex was kept, or every weight is 0, the
// single nearest live point of the body gets weight 1 (contact_mechanics.py:71-74);
// w /= sum(w); output[body][v] += w_v x forceBody (contact_mechanics.py:75-76).
//
// OUTPUT. `SkinPointForces`: per body, one force vector (N, body frame) per live point,
// the port of `touch.sensor_outputs` (touch.py:1598-1621: zeroed at the start of every
// `get_touch_obs`, then every contact spread once).
//
// SOLID. Single responsibility per type: the table holds rows, the spreader spreads. Open
// for extension: another spread rule is another `ContactSpreader`. Dependency inversion:
// the spreader takes a `SkinTable`, not the JSON file.
//
// WHY THE RESULT CAN DIFFER FROM PYTHON AT THE LAST DIGIT. MIMo caches the candidate list
// per (body, submesh, nearest vertex, search distance) (touch.py:1235-1238, `cachedmethod`)
// and the first search uses the 1.5 margin so that later contacts near the same vertex read
// the cache; the Swift search runs fresh every time with the same margin, so it equals a
// Python first search. The sum of the weights is formed in vertex order here and in
// search-pop order there; both are float64 sums of the same terms.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain); the "First build on a Mac" checklist is in TouchPooler.swift's header.

import CryptoKit
import Foundation

// MARK: - The skin table

/// One sensor mesh of a skin body: the vertices MIMo spread over one colliding geom
/// (touch.py:986-1024), moved into the body frame (touch.py:890). Row `submeshes[]` of a
/// body in the WI-13 JSON.
public struct SkinMeshRows: Decodable {
    /// Index of this submesh within its body (`submesh`).
    public let submesh: Int
    /// The MuJoCo geom id the mesh was spread over (`geom_id`).
    public let geomId: Int
    /// The geom's name (`geom_name`).
    public let geomName: String
    /// The geom's MuJoCo type name, such as "mjGEOM_CAPSULE" (`geom_type`).
    public let geomType: String
    /// The geom's three size numbers, metres (`geom_size_m`).
    public let geomSize: [Double]
    /// True for a trimesh PointCloud (no edges) (`is_point_cloud`).
    public let isPointCloud: Bool
    /// Every vertex's body-frame position, metres (`positions_m`), active and inactive.
    public let positions: [Vec3]
    /// `active[v]`: false when vertex v lies inside another geom of the same body.
    public let active: [Bool]
    /// `sensor_index[v]`: the vertex's index among the body's live points, -1 when inactive.
    public let sensorIndex: [Int]
    /// `neighbours[v]`: the vertex indices joined to v by a mesh edge, sorted.
    public let neighbours: [[Int]]

    /// The number of vertices (`vertex_count`).
    public var vertexCount: Int { positions.count }
    /// The number of active vertices (`active_count`).
    public var activeCount: Int { active.reduce(0) { $0 + ($1 ? 1 : 0) } }

    /// True when `active`, `sensorIndex` and `neighbours` have one entry per vertex and every
    /// neighbour index names a vertex of this submesh (the decoder checks the same).
    public var isConsistent: Bool {
        let n = positions.count
        guard active.count == n, sensorIndex.count == n, neighbours.count == n else { return false }
        return neighbours.allSatisfy { list in list.allSatisfy { $0 >= 0 && $0 < n } }
    }

    private enum CodingKeys: String, CodingKey {
        case submesh
        case geomId = "geom_id"
        case geomName = "geom_name"
        case geomType = "geom_type"
        case geomSize = "geom_size_m"
        case isPointCloud = "is_point_cloud"
        case vertexCount = "vertex_count"
        case activeCount = "active_count"
        case positions = "positions_m"
        case active
        case sensorIndex = "sensor_index"
        case neighbours
    }

    /// Builds a submesh by hand (tests and small tables). Every array must have one entry
    /// per vertex; `sensorIndex` is -1 for inactive vertices. A submesh that breaks this
    /// (see `isConsistent`) is refused by `SkinBodyRows.init?` and skipped by the search.
    public init(submesh: Int = 0, geomId: Int = -1, geomName: String = "", geomType: String = "",
                geomSize: [Double] = [0, 0, 0], isPointCloud: Bool = false,
                positions: [Vec3], active: [Bool], sensorIndex: [Int], neighbours: [[Int]]) {
        self.submesh = submesh
        self.geomId = geomId
        self.geomName = geomName
        self.geomType = geomType
        self.geomSize = geomSize
        self.isPointCloud = isPointCloud
        self.positions = positions
        self.active = active
        self.sensorIndex = sensorIndex
        self.neighbours = neighbours
    }

    /// Decodes one `submeshes[]` row of the WI-13 JSON and checks that every per-vertex
    /// array has `vertex_count` entries and that `active_count` matches the mask.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        submesh = try c.decode(Int.self, forKey: .submesh)
        geomId = try c.decode(Int.self, forKey: .geomId)
        geomName = try c.decode(String.self, forKey: .geomName)
        geomType = try c.decode(String.self, forKey: .geomType)
        geomSize = try c.decode([Double].self, forKey: .geomSize)
        isPointCloud = try c.decode(Bool.self, forKey: .isPointCloud)
        let declaredVertices = try c.decode(Int.self, forKey: .vertexCount)
        let declaredActive = try c.decode(Int.self, forKey: .activeCount)
        let rows = try c.decode([[Double]].self, forKey: .positions)
        var points: [Vec3] = []
        points.reserveCapacity(rows.count)
        for (v, row) in rows.enumerated() {
            guard let point = Vec3(array: row) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .positions, in: c,
                    debugDescription: "vertex \(v) has \(row.count) numbers, not 3")
            }
            points.append(point)
        }
        positions = points
        active = try c.decode([Bool].self, forKey: .active)
        sensorIndex = try c.decode([Int].self, forKey: .sensorIndex)
        neighbours = try c.decode([[Int]].self, forKey: .neighbours)
        let n = positions.count
        guard n == declaredVertices, active.count == n, sensorIndex.count == n, neighbours.count == n else {
            throw DecodingError.dataCorruptedError(
                forKey: .vertexCount, in: c,
                debugDescription: "submesh \(submesh) of geom \(geomId): vertex_count \(declaredVertices), "
                    + "positions \(n), active \(active.count), sensor_index \(sensorIndex.count), "
                    + "neighbours \(neighbours.count)")
        }
        guard activeCount == declaredActive else {
            throw DecodingError.dataCorruptedError(
                forKey: .activeCount, in: c,
                debugDescription: "submesh \(submesh) of geom \(geomId): active_count \(declaredActive) "
                    + "but the mask holds \(activeCount)")
        }
        for (v, list) in neighbours.enumerated() {
            for w in list where w < 0 || w >= n {
                throw DecodingError.dataCorruptedError(
                    forKey: .neighbours, in: c,
                    debugDescription: "submesh \(submesh): vertex \(v) names neighbour \(w) outside 0..<\(n)")
            }
        }
    }
}

/// One skin body: its scale, its submeshes and its live points in output order. Row
/// `bodies[]` of the WI-13 JSON; the port of `touch.sensor_positions[body]`,
/// `touch._submeshes[body]`, `touch._active_subvertices[body]` and
/// `touch._vertex_to_sensor_idx[body]` (touch.py:928-939).
public struct SkinBodyRows: Decodable {
    /// The MuJoCo body id (`body_id`).
    public let bodyId: Int
    /// The body's name (`body_name`), such as "right_ffdistal".
    public let bodyName: String
    /// The body's row in the pooler's patch sequence (`pooler_row`); -1 when unknown.
    public let poolerRow: Int
    /// The sensor spacing, metres (`scale_m`: touch_params scales x touch_scale_mult).
    public let scale: Double
    /// The spacing the per-contact search uses, metres (`force_spread_scale_m`;
    /// env.py:1170: sensor_scales / touch_scale_mult; equal to `scale` at multiplier 1.0).
    public let forceSpreadScale: Double
    /// The number of live points of the bodies before this one (`point_offset`).
    public let pointOffset: Int
    /// The submeshes, one per colliding geom, in add_body's order.
    public let submeshes: [SkinMeshRows]
    /// The live (active) vertices' positions in output order: submesh by submesh, vertex
    /// by vertex (`touch.sensor_positions[body]`, touch.py:939).
    public let livePositions: [Vec3]

    /// The number of live points (`sensor_count`).
    public var sensorCount: Int { livePositions.count }

    private enum CodingKeys: String, CodingKey {
        case bodyId = "body_id"
        case bodyName = "body_name"
        case poolerRow = "pooler_row"
        case scale = "scale_m"
        case forceSpreadScale = "force_spread_scale_m"
        case sensorCount = "sensor_count"
        case pointOffset = "point_offset"
        case submeshes
    }

    /// Builds a body by hand from its submeshes; the live positions are derived from the
    /// active masks in output order, and the sensor indices are checked to run 0, 1, 2 ...
    /// across the submeshes. Returns nil when a sensor index does not match its position in
    /// that order.
    public init?(bodyId: Int, bodyName: String, scale: Double, forceSpreadScale: Double? = nil,
                 poolerRow: Int = -1, pointOffset: Int = 0, submeshes: [SkinMeshRows]) {
        self.bodyId = bodyId
        self.bodyName = bodyName
        self.poolerRow = poolerRow
        self.scale = scale
        self.forceSpreadScale = forceSpreadScale ?? scale
        self.pointOffset = pointOffset
        self.submeshes = submeshes
        guard let live = SkinBodyRows.livePoints(of: submeshes) else { return nil }
        self.livePositions = live
    }

    /// Decodes one `bodies[]` row and checks `sensor_count` against the active vertices.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        bodyId = try c.decode(Int.self, forKey: .bodyId)
        bodyName = try c.decode(String.self, forKey: .bodyName)
        poolerRow = try c.decode(Int.self, forKey: .poolerRow)
        scale = try c.decode(Double.self, forKey: .scale)
        forceSpreadScale = try c.decode(Double.self, forKey: .forceSpreadScale)
        pointOffset = try c.decode(Int.self, forKey: .pointOffset)
        submeshes = try c.decode([SkinMeshRows].self, forKey: .submeshes)
        let declared = try c.decode(Int.self, forKey: .sensorCount)
        guard let live = SkinBodyRows.livePoints(of: submeshes) else {
            throw DecodingError.dataCorruptedError(
                forKey: .submeshes, in: c,
                debugDescription: "body \(bodyId) \(bodyName): sensor_index does not run 0, 1, 2 ... over the active vertices")
        }
        guard live.count == declared else {
            throw DecodingError.dataCorruptedError(
                forKey: .sensorCount, in: c,
                debugDescription: "body \(bodyId) \(bodyName): sensor_count \(declared) but \(live.count) active vertices")
        }
        livePositions = live
    }

    /// The active vertices' positions in output order, or nil when a submesh is not
    /// consistent or a vertex's sensor index differs from its position in that order
    /// (touch.py:930-934: cumsum(mask) - 1 plus the submesh offset).
    static func livePoints(of submeshes: [SkinMeshRows]) -> [Vec3]? {
        var live: [Vec3] = []
        for sub in submeshes {
            guard sub.isConsistent else { return nil }
            for v in 0..<sub.positions.count where sub.active[v] {
                if sub.sensorIndex[v] != live.count { return nil }
                live.append(sub.positions[v])
            }
        }
        return live
    }
}

/// A sensing body with no patch: the two eyes (`placeholder_bodies.bodies[]` of the WI-13
/// JSON; touch_adapter.py:68, :190: sensor spacing at or above 0.5 m).
public struct EyePlaceholderRow: Decodable {
    /// The MuJoCo body id.
    public let bodyId: Int
    /// The body's name.
    public let bodyName: String
    /// The sensor spacing, metres (1.0 for the eyes).
    public let scale: Double
    /// The number of points (1 for each eye).
    public let sensorCount: Int
    /// The points' body-frame positions, metres.
    public let positions: [[Double]]

    private enum CodingKeys: String, CodingKey {
        case bodyId = "body_id"
        case bodyName = "body_name"
        case scale = "scale_m"
        case sensorCount = "sensor_count"
        case positions = "positions_m"
    }
}

/// The `placeholder_bodies` object of the WI-13 JSON.
public struct EyePlaceholderRows: Decodable {
    /// The rule that puts a body here, as the exporter wrote it.
    public let rule: String
    /// The placeholder bodies in id order.
    public let bodies: [EyePlaceholderRow]
}

/// The skin points a contact can be spread over: the bodies in id order, found by id.
public protocol SkinTable {
    /// The skin bodies in MuJoCo body id order (the pooler's patch order).
    var bodies: [SkinBodyRows] { get }
    /// The body with this id, or nil when the table has no skin rows for it.
    func body(id: Int) -> SkinBodyRows?
    /// The number of live points over every body (4,683 for the shipped table).
    var pointCount: Int { get }
    /// The first 16 hex characters of the SHA-256 of the live positions' float64 bytes,
    /// bodies in id order (mimo_bridge/touch_inside_rays.py:126-131); 0581069917fae467 for
    /// the shipped table.
    var layoutDigest: String { get }
}

/// What can go wrong when a skin table is loaded.
public enum SkinTableError: Error, Equatable {
    /// The JSON's `format` is not "uema.skin_layout.v1".
    case unexpectedFormat(String)
    /// `point_count` or `body_count` disagrees with the rows.
    case countMismatch(String)
    /// The bodies are not in ascending id order.
    case bodiesOutOfOrder
}

/// The skin table apple/scripts/export_skin_layout.py writes (WI-13), decoded from its
/// JSON. Specification section 5: `ShippedSkinTable`.
public struct ShippedSkinTable: SkinTable, Decodable {
    /// The only JSON format this type reads.
    public static let expectedFormat = "uema.skin_layout.v1"

    /// The JSON's `format`.
    public let format: String
    /// `layout_digest` as written by the exporter.
    public let layoutDigest: String
    /// `point_count`: the live points over every body.
    public let pointCount: Int
    /// `vertex_count`: live and inactive vertices together.
    public let vertexCount: Int
    /// The skin bodies in id order.
    public let bodies: [SkinBodyRows]
    /// The sensing bodies without a patch (the eyes).
    public let placeholderBodies: EyePlaceholderRows
    /// Body id to its index in `bodies`.
    private let index: [Int: Int]

    private enum CodingKeys: String, CodingKey {
        case format
        case layoutDigest = "layout_digest"
        case pointCount = "point_count"
        case vertexCount = "vertex_count"
        case bodyCount = "body_count"
        case bodies
        case placeholderBodies = "placeholder_bodies"
    }

    /// Decodes the JSON document and refuses a wrong format, a body count or point count
    /// that disagrees with the rows, or bodies out of id order.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        format = try c.decode(String.self, forKey: .format)
        guard format == ShippedSkinTable.expectedFormat else {
            throw SkinTableError.unexpectedFormat(format)
        }
        layoutDigest = try c.decode(String.self, forKey: .layoutDigest)
        pointCount = try c.decode(Int.self, forKey: .pointCount)
        vertexCount = try c.decode(Int.self, forKey: .vertexCount)
        let bodyCount = try c.decode(Int.self, forKey: .bodyCount)
        bodies = try c.decode([SkinBodyRows].self, forKey: .bodies)
        placeholderBodies = try c.decode(EyePlaceholderRows.self, forKey: .placeholderBodies)
        guard bodies.count == bodyCount else {
            throw SkinTableError.countMismatch("body_count \(bodyCount), bodies \(bodies.count)")
        }
        let live = bodies.reduce(0) { $0 + $1.sensorCount }
        guard live == pointCount else {
            throw SkinTableError.countMismatch("point_count \(pointCount), live points \(live)")
        }
        let vertices = bodies.reduce(0) { total, body in
            total + body.submeshes.reduce(0) { $0 + $1.vertexCount }
        }
        guard vertices == vertexCount else {
            throw SkinTableError.countMismatch("vertex_count \(vertexCount), vertices \(vertices)")
        }
        var map: [Int: Int] = [:]
        var previous = Int.min
        for (i, body) in bodies.enumerated() {
            guard body.bodyId > previous else { throw SkinTableError.bodiesOutOfOrder }
            previous = body.bodyId
            map[body.bodyId] = i
        }
        index = map
    }

    /// Reads and decodes the JSON file at `url` (Foundation's JSONDecoder: "An object that
    /// decodes instances of a data type from JSON objects",
    /// https://developer.apple.com/documentation/foundation/jsondecoder).
    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url)
        self = try JSONDecoder().decode(ShippedSkinTable.self, from: data)
    }

    /// The body with this id, or nil (the eyes and every non-skin body).
    public func body(id: Int) -> SkinBodyRows? {
        guard let i = index[id] else { return nil }
        return bodies[i]
    }

    /// Recomputes the layout digest from the decoded positions: SHA-256 (CryptoKit SHA256,
    /// "An implementation of Secure Hashing Algorithm 2 (SHA-2) hashing with a 256-bit
    /// digest", https://developer.apple.com/documentation/cryptokit/sha256) over the live
    /// points' float64 little-endian bytes, bodies in id order, points in output order
    /// (touch_inside_rays.py:126-131; export_skin_layout.py:77-81), first 16 hex characters.
    /// Equal to `layoutDigest` when the JSON numbers decoded to the same float64 values the
    /// exporter wrote.
    public func recomputedDigest() -> String {
        var bytes = Data()
        bytes.reserveCapacity(pointCount * 24)
        for body in bodies {
            for point in body.livePositions {
                for component in [point.x, point.y, point.z] {
                    let bits = component.bitPattern.littleEndian
                    withUnsafeBytes(of: bits) { bytes.append(contentsOf: $0) }
                }
            }
        }
        let digest = SHA256.hash(data: bytes)
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return String(hex.prefix(16))
    }
}

// MARK: - Point forces

/// The force on every live skin point of every body, N, body frame: the port of
/// `touch.sensor_outputs` (touch.py:1563-1580 `get_empty_sensor_dict`, :1612 zeroed per
/// read). Keyed by body id; a body absent from the dictionary has no points.
public struct SkinPointForces: Equatable {
    /// Body id to one force per live point, in the table's output order.
    public private(set) var forces: [Int: [Vec3]]

    /// Zero forces for the given point counts per body.
    public init(pointCounts: [Int: Int]) {
        var map: [Int: [Vec3]] = [:]
        for (body, count) in pointCounts {
            map[body] = Array(repeating: Vec3.zero, count: max(0, count))
        }
        forces = map
    }

    /// Zero forces for every body of the table.
    public init(table: SkinTable) {
        var counts: [Int: Int] = [:]
        for body in table.bodies { counts[body.bodyId] = body.sensorCount }
        self.init(pointCounts: counts)
    }

    /// Adds `force` to point `point` of body `body`. Returns false, adding nothing, when
    /// the body is absent or the point index is out of range. MIMo stores the point forces
    /// as float32 (touch.py:1578 `np.zeros(..., dtype=np.float32)`) and numpy's in-place
    /// `+=` (contact_mechanics.py:76) adds in float64 and rounds the sum to float32, so each
    /// component of the sum is rounded to Float here before it is stored.
    @discardableResult
    public mutating func add(_ force: Vec3, body: Int, point: Int) -> Bool {
        guard var list = forces[body], point >= 0, point < list.count else { return false }
        let sum = list[point] + force
        list[point] = Vec3(Double(Float(sum.x)), Double(Float(sum.y)), Double(Float(sum.z)))
        forces[body] = list
        return true
    }

    /// Sets the force of point `point` of body `body` as given (no rounding); used to load
    /// recorded point forces such as the WI-14 fixture's. Returns false, setting nothing,
    /// when the body is absent or the point index is out of range.
    @discardableResult
    public mutating func set(_ force: Vec3, body: Int, point: Int) -> Bool {
        guard var list = forces[body], point >= 0, point < list.count else { return false }
        list[point] = force
        forces[body] = list
        return true
    }

    /// The forces of a body, or an empty array when the body is absent.
    public func forces(ofBody body: Int) -> [Vec3] {
        forces[body] ?? []
    }

    /// The number of points of a body whose force length exceeds `threshold` N
    /// (touch_adapter.py:233-234: |f| > ACTIVE_FORCE_EPS).
    public func activePointCount(ofBody body: Int, threshold: Double) -> Int {
        forces(ofBody: body).reduce(0) { $0 + ($1.norm > threshold ? 1 : 0) }
    }

    /// The vector sum of a body's point forces, N, body frame.
    public func totalForce(ofBody body: Int) -> Vec3 {
        forces(ofBody: body).reduce(Vec3.zero, +)
    }
}

// MARK: - The spreader

/// Spreads one contact's force over the skin points of its body.
public protocol ContactSpreader {
    /// Adds the contact's force, distributed by the spreader's rule, into `outputs`.
    func spread(_ contact: SkinContact, into outputs: inout SkinPointForces)
}

extension ContactSpreader {
    /// Spreads every contact into a fresh zero force table: the port of one
    /// `get_touch_obs` (touch.py:1598-1621).
    public func pointForces(for contacts: [SkinContact], table: SkinTable) -> SkinPointForces {
        var outputs = SkinPointForces(table: table)
        for contact in contacts { spread(contact, into: &outputs) }
        return outputs
    }
}

/// The bridge's `conservative_spread_linear` over MIMo's distance search
/// (specification section 5: `LinearContactSpreader`).
public struct LinearContactSpreader: ContactSpreader {
    /// Search radius as a multiple of the body's spread scale (contact_mechanics.py:69: 3).
    public static let searchMultiple: Double = 3.0
    /// Weight falls to 0 at this multiple of the spread scale (contact_mechanics.py:70: 2).
    public static let weightMultiple: Double = 2.0
    /// The widening of the first search (touch.py:1297-1298 `distance_margin=1.5`).
    public static let firstSearchMargin: Double = 1.5

    /// The skin table the spreader reads positions and neighbours from.
    public let table: SkinTable

    /// Makes a spreader over a table.
    public init(table: SkinTable) {
        self.table = table
    }

    /// The live points of `body` within `limit` metres of `position` (body frame) under
    /// MIMo's rule (file header, OPERATION), as (indices into the body's live points,
    /// their straight-line distances). Empty when the body is not in the table.
    public func sensorsWithinDistance(body bodyId: Int, position: Vec3, limit: Double)
        -> (indices: [Int], distances: [Double]) {
        guard let body = table.body(id: bodyId) else { return ([], []) }
        let searchDistance = limit * LinearContactSpreader.firstSearchMargin
        var indices: [Int] = []
        var distances: [Double] = []
        for sub in body.submeshes {
            let n = sub.positions.count
            if n == 0 || !sub.isConsistent { continue }
            var d: [Double] = []
            d.reserveCapacity(n)
            var nearest = 0
            for v in 0..<n {
                let distance = (sub.positions[v] - position).norm
                d.append(distance)
                if distance < d[nearest] { nearest = v }
            }
            // touch.py:1262-1265: the nearest vertex beyond the search distance skips the submesh
            if d[nearest] > searchDistance { continue }
            var candidates: [Int] = []
            if n == 1 {
                // touch.py:1268-1273
                if sub.active[0] { candidates.append(0) }
            } else {
                // touch.py:1276-1293: visit vertices within the search distance reachable
                // from the nearest vertex through such vertices
                var checked: [Bool] = d.map { !($0 < searchDistance) }
                var toCheck: [Int] = [nearest]
                toCheck.append(contentsOf: sub.neighbours[nearest])
                while let candidate = toCheck.popLast() {
                    if checked[candidate] { continue }
                    checked[candidate] = true
                    toCheck.append(contentsOf: sub.neighbours[candidate])
                    if sub.active[candidate] { candidates.append(candidate) }
                }
            }
            // touch.py:1352-1356: keep the candidates under the limit, as live point indices
            for v in candidates.sorted() where d[v] < limit {
                indices.append(sub.sensorIndex[v])
                distances.append(d[v])
            }
        }
        return (indices, distances)
    }

    /// contact_mechanics.py:66-76 for one contact. A body absent from the table (the eyes,
    /// whose single placeholder point no patch reads) gets nothing.
    public func spread(_ contact: SkinContact, into outputs: inout SkinPointForces) {
        guard let body = table.body(id: contact.bodyId) else { return }
        let scale = body.forceSpreadScale
        let position = contact.positionBody
        let found = sensorsWithinDistance(
            body: body.bodyId, position: position, limit: LinearContactSpreader.searchMultiple * scale)
        var indices = found.indices
        var weights = found.distances.map {
            max(1.0 - $0 / (LinearContactSpreader.weightMultiple * scale), 0.0)
        }
        var total = weights.reduce(0.0, +)
        if indices.isEmpty || total <= 0 {
            // contact_mechanics.py:71-74: the single nearest live point
            guard !body.livePositions.isEmpty else { return }
            var nearest = 0
            var nearestDistance = (body.livePositions[0] - position).norm
            for (i, point) in body.livePositions.enumerated() {
                let distance = (point - position).norm
                if distance < nearestDistance {
                    nearest = i
                    nearestDistance = distance
                }
            }
            indices = [nearest]
            weights = [1.0]
            total = 1.0
        }
        for (k, index) in indices.enumerated() {
            outputs.add(contact.forceBody * (weights[k] / total), body: body.bodyId, point: index)
        }
    }
}
