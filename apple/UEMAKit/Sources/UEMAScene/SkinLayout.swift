// SkinLayout.swift: the shipped skin layout, found in this module's resource bundle and parsed
// into the arrays the tactile reader searches (work item WI-26 of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-5 and the
// SkinTable row of its section 5).
//
// WHAT THIS FILE IS. The skin side of the UEMAScene module: SkinLayout decodes
// Resources/skin_layout_age24_trunk_v1.json, which apple/scripts/export_skin_layout.py (WI-13)
// wrote from one build of the released crib world (scene crib, body age 24 months, body layout
// trunk_v1, contact profile mechanical_v4, touch inside rays fixed_v1, touch scale multiplier
// 1.0: the export's WORLD constant). The scene side is SceneBundle.swift. Nothing in this file was
// compiled on the machine that wrote it (no Swift toolchain; plan section 3, fact 1).
//
// INPUT. The JSON document of format "uema.skin_layout.v1" (export_skin_layout.py,
// build_layout(), documents every key), 849,895 bytes at export on November 4, 2026:
// - bodies: 53 skin bodies in MuJoCo body id order (ids 2 to 56; the export's body_row()), each
//   with body_id, body_name, pooler_row (its row in the touch pooler, 0 to 52), scale_m (metres
//   between neighbouring sensor points: touch.sensor_scales), force_spread_scale_m (the scale of
//   the per-contact search: touch.force_spread_scales, equal to scale_m at multiplier 1.0),
//   sensor_count (its live points), point_offset (live points of the bodies before it) and
//   submeshes: one per colliding geom in MIMo's add_body order (touch.py:884-889), each with
//   submesh, geom_id, geom_name, geom_type (mjtGeom name), geom_size_m (3 numbers), is_point_cloud,
//   vertex_count, active_count, positions_m (every vertex, live and inactive, in the body frame
//   in metres: touch.py:890), active (the live mask: touch._active_subvertices), sensor_index
//   (the vertex's index in the body's output arrays, -1 when inactive: touch._vertex_to_sensor_idx)
//   and neighbours (each vertex's sorted neighbour vertex indices, the mesh's edges:
//   touch.py:1222-1235). The shipped file holds 73 submeshes with 5,005 vertices: 46 capsules,
//   12 boxes, 14 cylinders and 1 sphere, no point cloud (counted from the file on
//   November 4, 2026).
// - points: the 4,683 live points as rows [body_id, submesh, vertex, x_m, y_m, z_m] in digest
//   order (bodies in id order, then submesh order, then vertex order).
// - layout_digest: "0581069917fae467", the first 16 hex characters of the SHA-256 over the live
//   points' float64 [x, y, z] bytes in that order (mimo_bridge/touch_inside_rays.py:126-131); the
//   layout every released world was built with (touch_inside_rays.py:24-25).
// - placeholder_bodies: the two eye bodies (ids 45 and 46, one point each at a spacing of 1.0 m)
//   that form no patch (touch_adapter.py:68 SKIN_SCALE_MAX 0.5 m, :190) and stay outside the
//   digest.
// - world, search, units, versions, written_by, written_at, digest_rule: the provenance and the
//   search rule the tactile reader reproduces (search: distance limit 3 x force_spread_scale_m,
//   first-search margin 1.5, weight max(1 - distance / (2 x force_spread_scale_m), 0) normalised
//   to sum 1, else the single nearest live point: contact_mechanics.py:67-76,
//   touch.py:1297-1358).
//
// OPERATION. decode(_:) runs JSONDecoder over Codable types whose coding keys are the JSON's
// snake_case names; each points row is read from an unkeyed container as 3 Int then 3 Double;
// every 3-number list becomes a SIMD3<Double>. validate() checks the invariants the export's
// own test (apple/scripts/tests/test_export_skin_layout.py) checks on the file: body ids strictly
// increasing, pooler rows 0 to count-1, point offsets running, every submesh's arrays of one
// length with active_count live entries, sensor_index running over the live vertices and -1
// elsewhere, every points row equal to its vertex, neighbour lists sorted, in range, without
// self and symmetric, point and vertex totals equal to the header counts, the placeholder
// bodies disjoint from the skin bodies. computedDigest() rebuilds the digest from the decoded
// doubles (little-endian IEEE 754 bytes, the layout numpy's tobytes() writes on arm64) and
// checkShippedIdentity() compares count and digest with the shipped values.
//
// OUTPUT. bodies (the per-body tables, each with its submeshes), points, pointPositions (4,683
// SIMD3<Double> in digest order) and pointBodyIDs, and body(id:) for the tactile reader's
// per-contact search (WI-28: ContactReader, SkinSpread, TouchPooler), which uses: per body
// forceSpreadScaleM and sensorCount; per submesh positionsM, active, sensorIndex and
// neighbours (the breadth-first search of touch.py:1276-1293 walks inactive vertices too).
//
// SOURCES READ (all on November 4, 2026): apple/scripts/export_skin_layout.py entire;
// apple/scripts/tests/test_export_skin_layout.py entire; Source/Python/mimo_bridge/
// touch_inside_rays.py entire (skin_layout(), :126-131); Source/Python/mimo_bridge/
// contact_mechanics.py entire (conservative_spread_linear, :60-76); third_party/MIMo/mimo/
// mimo_infant/touch/touch.py:868-939 (add_body) and :1238-1358 (the within-distance search);
// mimo_bridge/touch_adapter.py:1-120 and :180-200 (SKIN_SCALE_MAX, the pooler's body order).
// Documentation pages (through developer.apple.com/tutorials/data/documentation/...):
// - https://developer.apple.com/documentation/foundation/jsondecoder/decode(_:from:) :
//   "func decode<T>(_ type: T.Type, from data: Data) throws -> T where T : Decodable".
// - https://developer.apple.com/documentation/cryptokit/hashfunction/hash(data:) : "static func
//   hash<D>(data: D) -> Self.Digest where D : DataProtocol" ("This can be any type that
//   conforms to DataProtocol, like Data or an array of UInt8 instances").
// - https://developer.apple.com/documentation/cryptokit/sha256digest : conforms to Sequence,
//   so the digest's bytes are read by iteration.
// - https://developer.apple.com/documentation/foundation/bundle/url(forresource:withextension:subdirectory:) :
//   returns nil when the resource is not found.
//
// FIRST BUILD ON A MAC (this work item's acceptance: "4,683 points parsed"; plan section 8,
// step 4). `xcodebuild -scheme UEMAKit-Package test -destination 'platform=macOS'
// -only-testing:UEMASceneTests` : exit code 0. Tests/UEMASceneTests/SkinLayoutTests.swift checks:
// the JSON is found; pointCount, points.count and pointPositions.count read 4,683; bodies.count
// 53; vertexCount 5,005; layoutDigest and computedDigest() both read "0581069917fae467"; the
// first and last points equal the file's values exactly; the hip, upper_body and left_big_toe
// rows equal the file's; validate() passes; a synthetic document decodes to its hand-computed
// digest; a malformed row and a wrong offset are refused by name. If computedDigest() differs
// from layoutDigest while every other test passes, JSONDecoder did not reproduce the exported
// doubles bit for bit: record the two digests in apple/FIRST_BUILD_RECORD.md and compare the
// first differing coordinate's decimal text with its decoded value before changing any code.
// The shipped layout serves this module's SkinLayoutReading protocol (renamed from SkinTable on
// November 4, 2026 to avoid the clash with UEMASenses.SkinTable; UEMACore declares no skin
// protocol). If a compile error in another module names `SkinTable` with a `bodies: [SkinBody]`
// member, that module was written against the earlier name: it reads `SkinLayoutReading` now.

import CryptoKit
import Foundation

/// Why a skin layout could not be located, decoded or trusted; every case names what failed.
public enum SkinLayoutError: Error, Equatable, CustomStringConvertible {
    /// `Bundle.url(forResource:withExtension:subdirectory:)` returned nil.
    case resourceNotFound(resource: String, subdirectory: String)
    /// The file could not be read; `reason` is the underlying error's text.
    case unreadable(path: String, reason: String)
    /// A value's shape is wrong at decode time (a row without 6 entries, a vector without 3).
    case malformed(String)
    /// One of validate()'s invariants does not hold; the text names the body and submesh.
    case invariant(String)
    /// The count or the digest differs from the shipped layout's.
    case identity(String)

    public var description: String {
        switch self {
        case let .resourceNotFound(resource, subdirectory):
            return "skin layout: \(resource) not found under \(subdirectory) in the module's resource bundle"
        case let .unreadable(path, reason):
            return "skin layout: \(path) unreadable: \(reason)"
        case let .malformed(text):
            return "skin layout: malformed: \(text)"
        case let .invariant(text):
            return "skin layout: invariant failed: \(text)"
        case let .identity(text):
            return "skin layout: identity failed: \(text)"
        }
    }
}

/// Reads a JSON list of exactly 3 numbers as a vector; used by the decoders below.
private func vector3(_ values: [Double], what: String) throws -> SIMD3<Double> {
    guard values.count == 3 else {
        throw SkinLayoutError.malformed("\(what) holds \(values.count) numbers, not 3")
    }
    return SIMD3<Double>(values[0], values[1], values[2])
}

/// One sensor mesh of a skin body: the mesh MIMo's TrimeshTouch spread over one colliding geom
/// (touch.py:868-939), every vertex of it, live or inactive, in the body frame.
public struct SkinSubmesh: Equatable {
    /// The submesh's index within its body (touch._submeshes[body_id] order).
    public let submesh: Int
    /// The MuJoCo geom id the mesh was spread over.
    public let geomID: Int
    /// The geom's name in the scene XML.
    public let geomName: String
    /// The geom's type as the mjtGeom enumerator name ("mjGEOM_CAPSULE", "mjGEOM_BOX",
    /// "mjGEOM_CYLINDER", "mjGEOM_SPHERE" occur in the shipped file).
    public let geomType: String
    /// The geom's size numbers in metres (MuJoCo's geom_size: 3 numbers, meaning per type).
    public let geomSizeM: SIMD3<Double>
    /// True when the mesh is a point cloud (no edges); false for every shipped submesh.
    public let isPointCloud: Bool
    /// The number of vertices, live and inactive.
    public let vertexCount: Int
    /// The number of live vertices (true entries of `active`).
    public let activeCount: Int
    /// Every vertex's position in the body frame, metres (touch.py:890 geom_pos_to_body).
    public let positionsM: [SIMD3<Double>]
    /// The live mask: false for a vertex inside another geom of the same body (touch.py:917-925).
    public let active: [Bool]
    /// The vertex's index in the body's output arrays, -1 when inactive.
    public let sensorIndex: [Int]
    /// Each vertex's neighbour vertex indices, sorted; the mesh's edges (touch.py:1222-1235).
    public let neighbours: [[Int]]

    /// Creates a submesh row (tests build synthetic layouts with it).
    public init(submesh: Int, geomID: Int, geomName: String, geomType: String,
                geomSizeM: SIMD3<Double>, isPointCloud: Bool, vertexCount: Int, activeCount: Int,
                positionsM: [SIMD3<Double>], active: [Bool], sensorIndex: [Int],
                neighbours: [[Int]]) {
        self.submesh = submesh
        self.geomID = geomID
        self.geomName = geomName
        self.geomType = geomType
        self.geomSizeM = geomSizeM
        self.isPointCloud = isPointCloud
        self.vertexCount = vertexCount
        self.activeCount = activeCount
        self.positionsM = positionsM
        self.active = active
        self.sensorIndex = sensorIndex
        self.neighbours = neighbours
    }
}

extension SkinSubmesh: Decodable {
    private enum CodingKeys: String, CodingKey {
        case submesh
        case geomID = "geom_id"
        case geomName = "geom_name"
        case geomType = "geom_type"
        case geomSizeM = "geom_size_m"
        case isPointCloud = "is_point_cloud"
        case vertexCount = "vertex_count"
        case activeCount = "active_count"
        case positionsM = "positions_m"
        case active
        case sensorIndex = "sensor_index"
        case neighbours
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let submesh = try container.decode(Int.self, forKey: .submesh)
        let geomID = try container.decode(Int.self, forKey: .geomID)
        let geomName = try container.decode(String.self, forKey: .geomName)
        let geomType = try container.decode(String.self, forKey: .geomType)
        let geomSizeM = try vector3(try container.decode([Double].self, forKey: .geomSizeM),
                                    what: "geom_size_m of geom \(geomID)")
        let isPointCloud = try container.decode(Bool.self, forKey: .isPointCloud)
        let vertexCount = try container.decode(Int.self, forKey: .vertexCount)
        let activeCount = try container.decode(Int.self, forKey: .activeCount)
        let rawPositions = try container.decode([[Double]].self, forKey: .positionsM)
        var positionsM: [SIMD3<Double>] = []
        positionsM.reserveCapacity(rawPositions.count)
        for (index, row) in rawPositions.enumerated() {
            positionsM.append(try vector3(row, what: "positions_m[\(index)] of geom \(geomID)"))
        }
        let active = try container.decode([Bool].self, forKey: .active)
        let sensorIndex = try container.decode([Int].self, forKey: .sensorIndex)
        let neighbours = try container.decode([[Int]].self, forKey: .neighbours)
        self.init(submesh: submesh, geomID: geomID, geomName: geomName, geomType: geomType,
                  geomSizeM: geomSizeM, isPointCloud: isPointCloud, vertexCount: vertexCount,
                  activeCount: activeCount, positionsM: positionsM, active: active,
                  sensorIndex: sensorIndex, neighbours: neighbours)
    }
}

/// One skin body: a MuJoCo body MIMo's touch senses whose sensor spacing is under
/// SKIN_SCALE_MAX 0.5 m (touch_adapter.py:68, :190), so it forms one patch of the touch pooler.
public struct SkinBody: Equatable, Decodable {
    /// The MuJoCo body id (2 to 56 in the shipped file).
    public let bodyID: Int
    /// The body's name in the scene XML ("hip", "left_hand", "head", ...).
    public let bodyName: String
    /// The body's row in the touch pooler, its position in body id order (0 to 52).
    public let poolerRow: Int
    /// Metres between neighbouring sensor points (touch.sensor_scales; touch_params scales x
    /// touch_scale_mult 1.0). Shipped values: 0.002 m to 0.038 m.
    public let scaleM: Double
    /// Metres, the scale of the per-contact search (touch.force_spread_scales): points within
    /// 3 x this distance are candidates, weight max(1 - distance / (2 x this), 0)
    /// (contact_mechanics.py:67-70). Equal to scaleM in the shipped file.
    public let forceSpreadScaleM: Double
    /// The body's live points (touch.get_sensor_count).
    public let sensorCount: Int
    /// Live points of the bodies before this one: this body's first row in `SkinLayout.points`.
    public let pointOffset: Int
    /// One submesh per colliding geom, in MIMo's add_body order.
    public let submeshes: [SkinSubmesh]

    private enum CodingKeys: String, CodingKey {
        case bodyID = "body_id"
        case bodyName = "body_name"
        case poolerRow = "pooler_row"
        case scaleM = "scale_m"
        case forceSpreadScaleM = "force_spread_scale_m"
        case sensorCount = "sensor_count"
        case pointOffset = "point_offset"
        case submeshes
    }

    /// Creates a body row (tests build synthetic layouts with it).
    public init(bodyID: Int, bodyName: String, poolerRow: Int, scaleM: Double,
                forceSpreadScaleM: Double, sensorCount: Int, pointOffset: Int,
                submeshes: [SkinSubmesh]) {
        self.bodyID = bodyID
        self.bodyName = bodyName
        self.poolerRow = poolerRow
        self.scaleM = scaleM
        self.forceSpreadScaleM = forceSpreadScaleM
        self.sensorCount = sensorCount
        self.pointOffset = pointOffset
        self.submeshes = submeshes
    }

    /// The live vertices' positions in output order (submesh order, then vertex order): MIMo's
    /// touch.sensor_positions[body_id], the array the single-nearest-point fallback searches
    /// (contact_mechanics.py:72-74).
    public var livePositions: [SIMD3<Double>] {
        var out: [SIMD3<Double>] = []
        out.reserveCapacity(sensorCount)
        for submesh in submeshes {
            for (index, position) in submesh.positionsM.enumerated()
            where index < submesh.active.count && submesh.active[index] {
                out.append(position)
            }
        }
        return out
    }
}

/// One live skin point as the file's points rows give it.
public struct SkinPoint: Equatable {
    /// The MuJoCo body id.
    public let bodyID: Int
    /// The submesh index within the body.
    public let submesh: Int
    /// The vertex index within the submesh.
    public let vertex: Int
    /// The position in the body frame, metres.
    public let position: SIMD3<Double>

    /// Creates a point row (tests build synthetic layouts with it).
    public init(bodyID: Int, submesh: Int, vertex: Int, position: SIMD3<Double>) {
        self.bodyID = bodyID
        self.submesh = submesh
        self.vertex = vertex
        self.position = position
    }
}

extension SkinPoint: Decodable {
    /// A row is the JSON list [body_id, submesh, vertex, x_m, y_m, z_m]: 3 integers, 3 doubles.
    public init(from decoder: Decoder) throws {
        var container = try decoder.unkeyedContainer()
        if let count = container.count, count != 6 {
            throw SkinLayoutError.malformed("a points row holds \(count) entries, not 6")
        }
        let bodyID = try container.decode(Int.self)
        let submesh = try container.decode(Int.self)
        let vertex = try container.decode(Int.self)
        let x = try container.decode(Double.self)
        let y = try container.decode(Double.self)
        let z = try container.decode(Double.self)
        self.init(bodyID: bodyID, submesh: submesh, vertex: vertex, position: SIMD3<Double>(x, y, z))
    }
}

/// A sensing body that forms no patch: the two eyes, one placeholder point each at a spacing
/// of 1.0 m, at or above SKIN_SCALE_MAX 0.5 m (touch_adapter.py:68, :190); outside the digest.
public struct SkinPlaceholderBody: Equatable {
    public let bodyID: Int
    public let bodyName: String
    public let scaleM: Double
    public let sensorCount: Int
    public let positionsM: [SIMD3<Double>]

    /// Creates a placeholder row (tests build synthetic layouts with it).
    public init(bodyID: Int, bodyName: String, scaleM: Double, sensorCount: Int,
                positionsM: [SIMD3<Double>]) {
        self.bodyID = bodyID
        self.bodyName = bodyName
        self.scaleM = scaleM
        self.sensorCount = sensorCount
        self.positionsM = positionsM
    }
}

extension SkinPlaceholderBody: Decodable {
    private enum CodingKeys: String, CodingKey {
        case bodyID = "body_id"
        case bodyName = "body_name"
        case scaleM = "scale_m"
        case sensorCount = "sensor_count"
        case positionsM = "positions_m"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let bodyID = try container.decode(Int.self, forKey: .bodyID)
        let bodyName = try container.decode(String.self, forKey: .bodyName)
        let scaleM = try container.decode(Double.self, forKey: .scaleM)
        let sensorCount = try container.decode(Int.self, forKey: .sensorCount)
        let rawPositions = try container.decode([[Double]].self, forKey: .positionsM)
        var positionsM: [SIMD3<Double>] = []
        positionsM.reserveCapacity(rawPositions.count)
        for (index, row) in rawPositions.enumerated() {
            positionsM.append(try vector3(row, what: "positions_m[\(index)] of placeholder body \(bodyID)"))
        }
        self.init(bodyID: bodyID, bodyName: bodyName, scaleM: scaleM, sensorCount: sensorCount,
                  positionsM: positionsM)
    }
}

/// The world the layout was built in (the file's "world" object; export_skin_layout.py WORLD
/// plus the build's report).
public struct SkinLayoutWorld: Equatable, Decodable {
    public let scene: String
    public let ageMonths: Double
    public let bodyLayout: String
    public let contactProfile: String
    public let touchInsideRays: String
    public let touchScaleMult: Double
    /// SHA-256 of the 10 fixed inside-test ray directions (touch_inside_rays.py FIXED_V1_SHA256).
    public let directionsSHA256: String
    /// Inside tests run through the switch during the build (58 in the shipped file).
    public let insideTests: Int
    /// Shape pairs MIMo's add_body tests (58 in the shipped file).
    public let shapePairs: Int
    /// The build world's nq (121: vision, caregiver, mobile and critters off for the build) and
    /// nbody (61); the released world has nq 146, and the skin is the same (the file's
    /// scene_additions text).
    public let nq: Int
    public let nbody: Int
    public let sceneXML: String
    public let sceneAdditions: String

    private enum CodingKeys: String, CodingKey {
        case scene
        case ageMonths = "age_months"
        case bodyLayout = "body_layout"
        case contactProfile = "contact_profile"
        case touchInsideRays = "touch_inside_rays"
        case touchScaleMult = "touch_scale_mult"
        case directionsSHA256 = "directions_sha256"
        case insideTests = "inside_tests"
        case shapePairs = "shape_pairs"
        case nq
        case nbody
        case sceneXML = "scene_xml"
        case sceneAdditions = "scene_additions"
    }
}

/// The per-contact search rule the file records (export_skin_layout.py SEARCH): the tactile
/// reader (WI-28) implements it; this type only carries the text and the one number.
public struct SkinLayoutSearch: Equatable, Decodable {
    public let scale: String
    public let distanceLimit: String
    /// 1.5: the factor the first search of a contact position widens its limit by
    /// (touch.get_sensors_within_distance distance_margin, touch.py:1297-1358).
    public let firstSearchMargin: Double
    public let weight: String
    public let source: String

    private enum CodingKeys: String, CodingKey {
        case scale
        case distanceLimit = "distance_limit"
        case firstSearchMargin = "first_search_margin"
        case weight
        case source
    }
}

/// What the tactile reader (WI-28: ContactReader, SkinSpread, TouchPooler) takes from the skin
/// layout (specification section 5, row "SkinTable ... ShippedSkinTable ... injected where:
/// TactileSense"): the per-body tables and the live points. A user is injected with a value of
/// this protocol type; SkinLayout is the shipped one.
///
/// UEMACore (Sources/UEMACore/Protocols.swift, WI-20) declares no skin protocol: its
/// `SceneLoading` hands out only the file URL (`skinLayoutURL`, Protocols.swift:471-472), so
/// SkinLayout conforms to no UEMACore protocol and this module's own protocol stays. It is not
/// named `SkinTable`: UEMASenses declares a public `SkinTable` of its own
/// (Sources/UEMASenses/Tactile/SkinSpread.swift:296, the row type its spreader searches, which
/// UEMASenses.SkinLayoutTable adapts SkinLayout into), and a file that imports both modules and
/// writes a name both declare gets an "ambiguous for type lookup" error.
public protocol SkinLayoutReading {
    /// The skin bodies in MuJoCo body id order, each with its submeshes (positions, live mask,
    /// output indices, neighbour lists) and its search scale.
    var bodies: [SkinBody] { get }
    /// The live point count (4,683 shipped).
    var pointCount: Int { get }
    /// The live points' positions in digest order, metres in the body frame.
    var pointPositions: [SIMD3<Double>] { get }
    /// The live points' body ids in digest order.
    var pointBodyIDs: [Int] { get }
    /// The layout digest the table states (0581069917fae467 shipped).
    var layoutDigest: String { get }
    /// The skin body with a MuJoCo body id, or nil when that body has no skin.
    func body(id: Int) -> SkinBody?
}

/// The skin layout: every skin body's sensor meshes and the 4,683 live points, as the tactile
/// reader needs them. One responsibility: locate, decode and check the shipped table; the
/// per-contact search over it is the tactile reader's (WI-28).
public struct SkinLayout: SkinLayoutReading, Equatable {
    /// The document format this type decodes (export_skin_layout.py FORMAT).
    public static let expectedFormat = "uema.skin_layout.v1"
    /// The shipped resource's file name without extension, under Resources/.
    public static let shippedResourceName = "skin_layout_age24_trunk_v1"
    /// The top-level folder the copy rule places in the resource bundle (Package.swift).
    public static let resourcesSubdirectory = "Resources"
    /// 4,683 live points at age 24 months with trunk_v1 (touch_inside_rays.py:24-25).
    public static let shippedPointCount = 4683
    /// The shipped layout's digest (touch_inside_rays.py:25).
    public static let shippedDigest = "0581069917fae467"
    /// 53 skin bodies (touch_adapter.py docstring: "53 skin bodies; the two eye bodies are not skin").
    public static let shippedBodyCount = 53

    public let format: String
    public let writtenBy: String
    public let writtenAt: String
    public let world: SkinLayoutWorld
    public let search: SkinLayoutSearch
    /// The file's own digest, 16 lowercase hex characters.
    public let layoutDigest: String
    /// The file's live point count.
    public let pointCount: Int
    /// The file's vertex count, live and inactive, over every submesh.
    public let vertexCount: Int
    /// The file's skin body count.
    public let bodyCount: Int
    /// The skin bodies in MuJoCo body id order.
    public let bodies: [SkinBody]
    /// The live points in digest order.
    public let points: [SkinPoint]
    /// The rule text for the placeholder bodies.
    public let placeholderRule: String
    /// The sensing bodies that form no patch (the two eyes).
    public let placeholderBodies: [SkinPlaceholderBody]

    /// The live points' positions in digest order, one SIMD3<Double> per point (metres, body frame).
    public let pointPositions: [SIMD3<Double>]
    /// The live points' body ids in digest order.
    public let pointBodyIDs: [Int]

    private let bodyIndexByID: [Int: Int]

    /// Creates a layout from decoded parts (tests build synthetic layouts with it); derives
    /// `pointPositions`, `pointBodyIDs` and the body index.
    public init(format: String, writtenBy: String, writtenAt: String, world: SkinLayoutWorld,
                search: SkinLayoutSearch, layoutDigest: String, pointCount: Int,
                vertexCount: Int, bodyCount: Int, bodies: [SkinBody], points: [SkinPoint],
                placeholderRule: String, placeholderBodies: [SkinPlaceholderBody]) {
        self.format = format
        self.writtenBy = writtenBy
        self.writtenAt = writtenAt
        self.world = world
        self.search = search
        self.layoutDigest = layoutDigest
        self.pointCount = pointCount
        self.vertexCount = vertexCount
        self.bodyCount = bodyCount
        self.bodies = bodies
        self.points = points
        self.placeholderRule = placeholderRule
        self.placeholderBodies = placeholderBodies
        self.pointPositions = points.map { $0.position }
        self.pointBodyIDs = points.map { $0.bodyID }
        var index: [Int: Int] = [:]
        for (position, body) in bodies.enumerated() {
            index[body.bodyID] = position
        }
        self.bodyIndexByID = index
    }

    /// The skin body with a MuJoCo body id, or nil when the id is not a skin body (the eyes, the
    /// mobile, the caregiver and every other body without skin).
    public func body(id: Int) -> SkinBody? {
        guard let position = bodyIndexByID[id] else { return nil }
        return bodies[position]
    }

    /// The digest rule of mimo_bridge/touch_inside_rays.py:126-131 on the decoded points: the
    /// SHA-256 over every live point's x, y, z as little-endian IEEE 754 doubles (24 bytes per
    /// point) in points order, first 16 hex characters. Equal to `layoutDigest` when the decoder
    /// reproduced the exported doubles bit for bit and the rows are in digest order.
    public func computedDigest() -> String {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(points.count * 24)
        for point in points {
            for coordinate in [point.position.x, point.position.y, point.position.z] {
                let pattern = coordinate.bitPattern.littleEndian
                withUnsafeBytes(of: pattern) { raw in
                    bytes.append(contentsOf: raw)
                }
            }
        }
        let digest = SHA256.hash(data: bytes)
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return String(hex.prefix(16))
    }

    /// Checks the invariants the export's own test checks on the file
    /// (apple/scripts/tests/test_export_skin_layout.py; the rules are listed in this file's
    /// header). Cost: one pass over the 5,005 vertices and their neighbour lists.
    /// - Throws: `SkinLayoutError.invariant` naming the first body, submesh or row that breaks a rule.
    public func validate() throws {
        guard format == SkinLayout.expectedFormat else {
            throw SkinLayoutError.invariant("format is \(format), not \(SkinLayout.expectedFormat)")
        }
        guard bodies.count == bodyCount else {
            throw SkinLayoutError.invariant("bodies holds \(bodies.count) rows, body_count says \(bodyCount)")
        }
        guard points.count == pointCount else {
            throw SkinLayoutError.invariant("points holds \(points.count) rows, point_count says \(pointCount)")
        }
        let poolerRows = bodies.map { $0.poolerRow }.sorted()
        guard poolerRows == Array(0..<bodies.count) else {
            throw SkinLayoutError.invariant("pooler rows \(poolerRows) are not 0 to \(bodies.count - 1)")
        }
        var previousID = Int.min
        var offset = 0
        var vertexTotal = 0
        var row = 0
        for body in bodies {
            guard body.bodyID > previousID else {
                throw SkinLayoutError.invariant("body \(body.bodyID) follows body \(previousID): ids not strictly increasing")
            }
            previousID = body.bodyID
            guard body.pointOffset == offset else {
                throw SkinLayoutError.invariant("body \(body.bodyID) point_offset \(body.pointOffset), expected \(offset)")
            }
            guard body.scaleM > 0, body.forceSpreadScaleM > 0 else {
                throw SkinLayoutError.invariant("body \(body.bodyID) scale_m \(body.scaleM) or force_spread_scale_m \(body.forceSpreadScaleM) not positive")
            }
            var liveInBody = 0
            for (submeshIndex, submesh) in body.submeshes.enumerated() {
                let n = submesh.vertexCount
                let where_ = "body \(body.bodyID) submesh \(submeshIndex)"
                guard submesh.submesh == submeshIndex else {
                    throw SkinLayoutError.invariant("\(where_) carries submesh index \(submesh.submesh)")
                }
                guard submesh.positionsM.count == n, submesh.active.count == n,
                      submesh.sensorIndex.count == n, submesh.neighbours.count == n else {
                    throw SkinLayoutError.invariant("\(where_) arrays are not all of length vertex_count \(n): positions \(submesh.positionsM.count), active \(submesh.active.count), sensor_index \(submesh.sensorIndex.count), neighbours \(submesh.neighbours.count)")
                }
                let liveHere = submesh.active.filter { $0 }.count
                guard liveHere == submesh.activeCount else {
                    throw SkinLayoutError.invariant("\(where_) active_count \(submesh.activeCount), active holds \(liveHere) true entries")
                }
                for vertex in 0..<n {
                    if submesh.active[vertex] {
                        guard submesh.sensorIndex[vertex] == liveInBody else {
                            throw SkinLayoutError.invariant("\(where_) vertex \(vertex) sensor_index \(submesh.sensorIndex[vertex]), expected \(liveInBody)")
                        }
                        guard row < points.count else {
                            throw SkinLayoutError.invariant("\(where_) vertex \(vertex) has no points row (rows exhausted at \(row))")
                        }
                        let point = points[row]
                        guard point.bodyID == body.bodyID, point.submesh == submeshIndex,
                              point.vertex == vertex else {
                            throw SkinLayoutError.invariant("points row \(row) is [\(point.bodyID), \(point.submesh), \(point.vertex)], expected [\(body.bodyID), \(submeshIndex), \(vertex)]")
                        }
                        guard point.position == submesh.positionsM[vertex] else {
                            throw SkinLayoutError.invariant("points row \(row) position differs from \(where_) vertex \(vertex)")
                        }
                        row += 1
                        liveInBody += 1
                    } else {
                        guard submesh.sensorIndex[vertex] == -1 else {
                            throw SkinLayoutError.invariant("\(where_) inactive vertex \(vertex) sensor_index \(submesh.sensorIndex[vertex]), expected -1")
                        }
                    }
                    let list = submesh.neighbours[vertex]
                    var previous = Int.min
                    for neighbour in list {
                        guard neighbour > previous else {
                            throw SkinLayoutError.invariant("\(where_) vertex \(vertex) neighbours not sorted and unique: \(list)")
                        }
                        previous = neighbour
                        guard neighbour != vertex else {
                            throw SkinLayoutError.invariant("\(where_) vertex \(vertex) lists itself as a neighbour")
                        }
                        guard neighbour >= 0, neighbour < n else {
                            throw SkinLayoutError.invariant("\(where_) vertex \(vertex) neighbour \(neighbour) outside 0..<\(n)")
                        }
                        guard submesh.neighbours[neighbour].contains(vertex) else {
                            throw SkinLayoutError.invariant("\(where_) edge \(vertex)-\(neighbour) is not symmetric")
                        }
                    }
                    if submesh.isPointCloud || n == 1 {
                        guard list.isEmpty else {
                            throw SkinLayoutError.invariant("\(where_) is a point cloud or one vertex yet vertex \(vertex) has neighbours")
                        }
                    }
                }
                vertexTotal += n
            }
            guard liveInBody == body.sensorCount else {
                throw SkinLayoutError.invariant("body \(body.bodyID) sensor_count \(body.sensorCount), submeshes hold \(liveInBody) live vertices")
            }
            offset += liveInBody
        }
        guard row == points.count else {
            throw SkinLayoutError.invariant("\(points.count - row) points rows remain after the last body")
        }
        guard offset == pointCount else {
            throw SkinLayoutError.invariant("bodies hold \(offset) live points, point_count says \(pointCount)")
        }
        guard vertexTotal == vertexCount else {
            throw SkinLayoutError.invariant("submeshes hold \(vertexTotal) vertices, vertex_count says \(vertexCount)")
        }
        let skinIDs = Set(bodies.map { $0.bodyID })
        for placeholder in placeholderBodies {
            guard !skinIDs.contains(placeholder.bodyID) else {
                throw SkinLayoutError.invariant("placeholder body \(placeholder.bodyID) is also a skin body")
            }
            guard placeholder.sensorCount == placeholder.positionsM.count else {
                throw SkinLayoutError.invariant("placeholder body \(placeholder.bodyID) sensor_count \(placeholder.sensorCount), positions \(placeholder.positionsM.count)")
            }
        }
    }

    /// The shipped identity: point count 4,683 and digest 0581069917fae467, both as the file
    /// states them and as recomputed from the decoded doubles (REQ-B-5: "the shipped table's
    /// digest equals 0581069917fae467").
    /// - Throws: `SkinLayoutError.identity` naming the value that differs.
    public func checkShippedIdentity() throws {
        guard pointCount == SkinLayout.shippedPointCount, points.count == SkinLayout.shippedPointCount else {
            throw SkinLayoutError.identity("point count \(pointCount) (\(points.count) rows), shipped \(SkinLayout.shippedPointCount)")
        }
        guard layoutDigest == SkinLayout.shippedDigest else {
            throw SkinLayoutError.identity("layout_digest \(layoutDigest), shipped \(SkinLayout.shippedDigest)")
        }
        let computed = computedDigest()
        guard computed == SkinLayout.shippedDigest else {
            throw SkinLayoutError.identity("digest recomputed from the decoded points \(computed), shipped \(SkinLayout.shippedDigest)")
        }
    }

    /// Decodes a layout document from its JSON bytes.
    /// - Throws: `SkinLayoutError.malformed` for a wrong row or vector length; the decoder's
    ///   `DecodingError` for a missing key or a wrong value type.
    public static func decode(_ data: Data) throws -> SkinLayout {
        try JSONDecoder().decode(SkinLayout.self, from: data)
    }

    /// Reads and decodes the layout at a file URL.
    /// - Throws: `SkinLayoutError.unreadable` when the file cannot be read; decode's errors.
    public static func load(from url: URL) throws -> SkinLayout {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw SkinLayoutError.unreadable(path: url.path, reason: String(describing: error))
        }
        return try decode(data)
    }

    /// The shipped resource's URL in a resource bundle: Resources/skin_layout_age24_trunk_v1.json.
    /// - Throws: `SkinLayoutError.resourceNotFound` when the bundle has no such file.
    public static func shippedURL(in bundle: Bundle = UEMASceneModule.resourceBundle) throws -> URL {
        guard let url = bundle.url(forResource: shippedResourceName, withExtension: "json",
                                   subdirectory: resourcesSubdirectory) else {
            throw SkinLayoutError.resourceNotFound(resource: "\(shippedResourceName).json",
                                                   subdirectory: resourcesSubdirectory)
        }
        return url
    }

    /// Locates, reads and decodes the shipped layout. Cost: one read of 849,895 bytes (the file's
    /// size at export on November 4, 2026) and one JSON decode.
    public static func loadShipped(from bundle: Bundle = UEMASceneModule.resourceBundle) throws -> SkinLayout {
        try load(from: try shippedURL(in: bundle))
    }
}

extension SkinLayout: Decodable {
    private enum CodingKeys: String, CodingKey {
        case format
        case writtenBy = "written_by"
        case writtenAt = "written_at"
        case world
        case search
        case layoutDigest = "layout_digest"
        case pointCount = "point_count"
        case vertexCount = "vertex_count"
        case bodyCount = "body_count"
        case bodies
        case points
        case placeholderBodies = "placeholder_bodies"
    }

    private enum PlaceholderKeys: String, CodingKey {
        case rule
        case bodies
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let placeholders = try container.nestedContainer(keyedBy: PlaceholderKeys.self,
                                                         forKey: .placeholderBodies)
        self.init(
            format: try container.decode(String.self, forKey: .format),
            writtenBy: try container.decode(String.self, forKey: .writtenBy),
            writtenAt: try container.decode(String.self, forKey: .writtenAt),
            world: try container.decode(SkinLayoutWorld.self, forKey: .world),
            search: try container.decode(SkinLayoutSearch.self, forKey: .search),
            layoutDigest: try container.decode(String.self, forKey: .layoutDigest),
            pointCount: try container.decode(Int.self, forKey: .pointCount),
            vertexCount: try container.decode(Int.self, forKey: .vertexCount),
            bodyCount: try container.decode(Int.self, forKey: .bodyCount),
            bodies: try container.decode([SkinBody].self, forKey: .bodies),
            points: try container.decode([SkinPoint].self, forKey: .points),
            placeholderRule: try placeholders.decode(String.self, forKey: .rule),
            placeholderBodies: try placeholders.decode([SkinPlaceholderBody].self, forKey: .bodies))
    }
}
