// TactileSkinSpreadTests.swift: hand-computed checks of the contact spread (work item
// WI-28; acceptance "one synthetic contact spreads to hand-computed weights").
//
// The worked example (the rule of contact_mechanics.py:67-76 over the search of
// touch.py:1238-1358): one body, spacing 0.01 m, three live points on a line at x = 0,
// 0.01 and 0.015 m joined 0-1-2; a contact at the origin with force (0, 0, 10) N.
//   search limit  = 3 x 0.01 = 0.03 m: all three points are in range
//   weights       = max(1 - d / 0.02, 0) = 1, 0.5, 0.25; sum 1.75
//   normalised    = 4/7, 2/7, 1/7
//   point forces  = (0, 0, 40/7), (0, 0, 20/7), (0, 0, 10/7) N; their sum is (0, 0, 10) N.
// The point forces are stored as float32 (touch.py:1578), so force checks use 1e-6 N: float32
// resolution at 10 N is 9.5e-7 N (2^-20).
// Nothing in this file was compiled on the machine that wrote it (no Swift toolchain).

import Foundation
import XCTest
import UEMASenses

final class TactileSkinSpreadTests: XCTestCase {
    /// A single-submesh body with the given points, all active, chained by `edges`.
    private func body(id: Int, scale: Double, points: [Vec3], active: [Bool]? = nil,
                      edges: [(Int, Int)]) -> SkinBodyRows? {
        let n = points.count
        var neighbours = [[Int]](repeating: [], count: n)
        for (a, b) in edges {
            neighbours[a].append(b)
            neighbours[b].append(a)
        }
        let mask = active ?? [Bool](repeating: true, count: n)
        var sensorIndex: [Int] = []
        var running = 0
        for v in 0..<n {
            if mask[v] {
                sensorIndex.append(running)
                running += 1
            } else {
                sensorIndex.append(-1)
            }
        }
        let mesh = SkinMeshRows(positions: points, active: mask, sensorIndex: sensorIndex,
                                neighbours: neighbours.map { $0.sorted() })
        return SkinBodyRows(bodyId: id, bodyName: "body\(id)", scale: scale, submeshes: [mesh])
    }

    /// A table of hand-built bodies.
    private struct HandTable: SkinTable {
        let bodies: [SkinBodyRows]
        func body(id: Int) -> SkinBodyRows? { bodies.first { $0.bodyId == id } }
        var pointCount: Int { bodies.reduce(0) { $0 + $1.sensorCount } }
        var layoutDigest: String { "hand" }
    }

    /// A contact on `body` at `position` (body frame) carrying `force` (body frame).
    private func contact(body: Int, position: Vec3, force: Vec3) -> SkinContact {
        SkinContact(bodyId: body, geom1: 1, geom2: 2, normalForce: force.norm,
                    tangentialBody: .zero, positionBody: position, positionWorld: position,
                    normalWorld: Vec3(0, 0, 1), normalBody: Vec3(0, 0, 1), forceBody: force,
                    penetration: 0, closingSpeed: 0, tangentSpeed: 0, frictionPower: 0,
                    otherBodyName: "world", otherIsSelf: false)
    }

    /// The worked example: weights 4/7, 2/7, 1/7 and the force conserved.
    func testOneContactSpreadsToHandComputedWeights() throws {
        let line = try XCTUnwrap(body(id: 7, scale: 0.01,
                                      points: [Vec3(0, 0, 0), Vec3(0.01, 0, 0), Vec3(0.015, 0, 0)],
                                      edges: [(0, 1), (1, 2)]))
        let table = HandTable(bodies: [line])
        let spreader = LinearContactSpreader(table: table)
        let found = spreader.sensorsWithinDistance(body: 7, position: .zero, limit: 0.03)
        XCTAssertEqual(found.indices, [0, 1, 2])
        XCTAssertEqual(found.distances.count, 3)
        XCTAssertEqual(found.distances[1], 0.01, accuracy: 1e-6)
        var outputs = SkinPointForces(table: table)
        spreader.spread(contact(body: 7, position: .zero, force: Vec3(0, 0, 10)), into: &outputs)
        let forces = outputs.forces(ofBody: 7)
        XCTAssertEqual(forces.count, 3)
        XCTAssertEqual(forces[0].z, 40.0 / 7.0, accuracy: 1e-6)
        XCTAssertEqual(forces[1].z, 20.0 / 7.0, accuracy: 1e-6)
        XCTAssertEqual(forces[2].z, 10.0 / 7.0, accuracy: 1e-6)
        XCTAssertEqual(forces[0].x, 0, accuracy: 0)
        let total = outputs.totalForce(ofBody: 7)
        XCTAssertEqual(total.z, 10.0, accuracy: 1e-6)
    }

    /// The force is conserved in every direction (test_contact_mechanics.py:41-48).
    func testEveryDirectionIsConserved() throws {
        let line = try XCTUnwrap(body(id: 7, scale: 0.01,
                                      points: [Vec3(0, 0, 0), Vec3(0.01, 0, 0), Vec3(0.015, 0, 0)],
                                      edges: [(0, 1), (1, 2)]))
        let table = HandTable(bodies: [line])
        let spreader = LinearContactSpreader(table: table)
        for force in [Vec3(0, 0, 10), Vec3(0, 7, 0), Vec3(5, 0, 0), Vec3(-3, 4, 5)] {
            var outputs = SkinPointForces(table: table)
            spreader.spread(contact(body: 7, position: .zero, force: force), into: &outputs)
            let total = outputs.totalForce(ofBody: 7)
            XCTAssertEqual(total.x, force.x, accuracy: 1e-6)
            XCTAssertEqual(total.y, force.y, accuracy: 1e-6)
            XCTAssertEqual(total.z, force.z, accuracy: 1e-6)
        }
    }

    /// Every point in range at 2 x scale or farther has weight 0: the single nearest live
    /// point takes the whole force (contact_mechanics.py:71-74).
    func testAllZeroWeightsFallBackToTheNearestPoint() throws {
        let line = try XCTUnwrap(body(id: 7, scale: 0.01,
                                      points: [Vec3(0.02, 0, 0), Vec3(0.025, 0, 0)],
                                      edges: [(0, 1)]))
        let table = HandTable(bodies: [line])
        let spreader = LinearContactSpreader(table: table)
        var outputs = SkinPointForces(table: table)
        spreader.spread(contact(body: 7, position: .zero, force: Vec3(0, 0, 10)), into: &outputs)
        let forces = outputs.forces(ofBody: 7)
        XCTAssertEqual(forces[0].z, 10.0, accuracy: 1e-6)
        XCTAssertEqual(forces[1].z, 0.0, accuracy: 0)
    }

    /// A contact farther than 3 x scale from every point still lands on the nearest point.
    func testOutOfRangeContactLandsOnTheNearestPoint() throws {
        let line = try XCTUnwrap(body(id: 7, scale: 0.01,
                                      points: [Vec3(0.1, 0, 0), Vec3(0.11, 0, 0)],
                                      edges: [(0, 1)]))
        let table = HandTable(bodies: [line])
        let spreader = LinearContactSpreader(table: table)
        XCTAssertTrue(spreader.sensorsWithinDistance(body: 7, position: .zero, limit: 0.03).indices.isEmpty)
        var outputs = SkinPointForces(table: table)
        spreader.spread(contact(body: 7, position: .zero, force: Vec3(1, 2, 3)), into: &outputs)
        let forces = outputs.forces(ofBody: 7)
        XCTAssertEqual(forces[0].y, 2.0, accuracy: 1e-6)
        XCTAssertEqual(forces[1].norm, 0.0, accuracy: 0)
    }

    /// A vertex in range but not reachable through the neighbour lists is excluded
    /// (touch.py:1276-1293: the search walks edges from the nearest vertex).
    func testUnreachableVertexIsExcluded() throws {
        let points = [Vec3(0, 0, 0), Vec3(0.01, 0, 0), Vec3(0, 0.01, 0)]
        let line = try XCTUnwrap(body(id: 7, scale: 0.01, points: points, edges: [(0, 1)]))
        let table = HandTable(bodies: [line])
        let spreader = LinearContactSpreader(table: table)
        let found = spreader.sensorsWithinDistance(body: 7, position: .zero, limit: 0.03)
        XCTAssertEqual(found.indices, [0, 1])
    }

    /// An inactive vertex is walked through but never collected (touch.py:1290-1291).
    func testInactiveVertexIsWalkedButNotCollected() throws {
        let points = [Vec3(0, 0, 0), Vec3(0.01, 0, 0), Vec3(0.02, 0, 0)]
        let line = try XCTUnwrap(body(id: 7, scale: 0.01, points: points,
                                      active: [true, false, true], edges: [(0, 1), (1, 2)]))
        XCTAssertEqual(line.sensorCount, 2)
        let table = HandTable(bodies: [line])
        let spreader = LinearContactSpreader(table: table)
        let found = spreader.sensorsWithinDistance(body: 7, position: .zero, limit: 0.03)
        XCTAssertEqual(found.indices, [0, 1])
        XCTAssertEqual(found.distances[1], 0.02, accuracy: 1e-6)
    }

    /// The first-search margin 1.5 (touch.py:1343): the walk may pass through a vertex at
    /// 3.5 x scale (under 4.5 x scale) to reach one at 2.5 x scale, which is then kept
    /// (under 3 x scale) while the bridge vertex itself is not.
    func testSearchMarginBridgesThroughAFarVertex() throws {
        let points = [Vec3(0, 0, 0), Vec3(0.035, 0, 0), Vec3(0.035, 0.0, 0.0) - Vec3(0.01, 0, 0)]
        // points: 0 at the origin, 1 at 0.035 m (the bridge), 2 at 0.025 m, edges 0-1 and 1-2
        let line = try XCTUnwrap(body(id: 7, scale: 0.01, points: points, edges: [(0, 1), (1, 2)]))
        let table = HandTable(bodies: [line])
        let spreader = LinearContactSpreader(table: table)
        let found = spreader.sensorsWithinDistance(body: 7, position: .zero, limit: 0.03)
        XCTAssertEqual(found.indices, [0, 2])
        // a bridge at 5 x scale is outside the search distance: vertex 2 becomes unreachable
        let far = [Vec3(0, 0, 0), Vec3(0.05, 0, 0), Vec3(0.025, 0, 0)]
        let line2 = try XCTUnwrap(body(id: 7, scale: 0.01, points: far, edges: [(0, 1), (1, 2)]))
        let spreader2 = LinearContactSpreader(table: HandTable(bodies: [line2]))
        XCTAssertEqual(spreader2.sensorsWithinDistance(body: 7, position: .zero, limit: 0.03).indices, [0])
    }

    /// Two submeshes are searched from their own nearest vertices (touch.py:1256-1259).
    func testEachSubmeshIsSearchedFromItsOwnNearestVertex() throws {
        let meshA = SkinMeshRows(positions: [Vec3(0, 0, 0), Vec3(0.01, 0, 0)], active: [true, true],
                                 sensorIndex: [0, 1], neighbours: [[1], [0]])
        let meshB = SkinMeshRows(positions: [Vec3(0, 0.012, 0)], active: [true],
                                 sensorIndex: [2], neighbours: [[]])
        let body = try XCTUnwrap(SkinBodyRows(bodyId: 9, bodyName: "two", scale: 0.01,
                                              submeshes: [meshA, meshB]))
        XCTAssertEqual(body.sensorCount, 3)
        let spreader = LinearContactSpreader(table: HandTable(bodies: [body]))
        let found = spreader.sensorsWithinDistance(body: 9, position: .zero, limit: 0.03)
        XCTAssertEqual(found.indices, [0, 1, 2])
        XCTAssertEqual(found.distances[2], 0.012, accuracy: 1e-6)
    }

    /// A body absent from the table (an eye) receives nothing and raises nothing.
    func testUnknownBodyIsSkipped() throws {
        let line = try XCTUnwrap(body(id: 7, scale: 0.01, points: [Vec3(0, 0, 0)], edges: []))
        let table = HandTable(bodies: [line])
        var outputs = SkinPointForces(table: table)
        LinearContactSpreader(table: table).spread(contact(body: 45, position: .zero, force: Vec3(1, 0, 0)), into: &outputs)
        XCTAssertEqual(outputs.totalForce(ofBody: 7).norm, 0, accuracy: 0)
        XCTAssertTrue(outputs.forces(ofBody: 45).isEmpty)
    }

    /// A hand-built submesh whose arrays disagree with its vertex count, or whose neighbour
    /// list names a vertex it does not have, is refused (no index trap in the search).
    func testInconsistentSubmeshIsRefused() {
        let short = SkinMeshRows(positions: [Vec3(0, 0, 0), Vec3(0.01, 0, 0)], active: [true],
                                 sensorIndex: [0, 1], neighbours: [[1], [0]])
        XCTAssertFalse(short.isConsistent)
        XCTAssertNil(SkinBodyRows(bodyId: 1, bodyName: "short", scale: 0.01, submeshes: [short]))
        let farNeighbour = SkinMeshRows(positions: [Vec3(0, 0, 0), Vec3(0.01, 0, 0)], active: [true, true],
                                        sensorIndex: [0, 1], neighbours: [[5], [0]])
        XCTAssertFalse(farNeighbour.isConsistent)
        XCTAssertNil(SkinBodyRows(bodyId: 1, bodyName: "far", scale: 0.01, submeshes: [farNeighbour]))
    }

    /// A sensor index that does not run 0, 1, 2 ... over the active vertices is refused.
    func testBodyWithWrongSensorIndexIsRefused() {
        let mesh = SkinMeshRows(positions: [Vec3(0, 0, 0), Vec3(0.01, 0, 0)], active: [true, true],
                                sensorIndex: [1, 0], neighbours: [[1], [0]])
        XCTAssertNil(SkinBodyRows(bodyId: 1, bodyName: "bad", scale: 0.01, submeshes: [mesh]))
    }
}
