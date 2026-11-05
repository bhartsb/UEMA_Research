// SkinLayoutTests.swift: the tests of SkinLayout (work item WI-26; specification REQ-B-5's
// "the shipped table's digest equals 0581069917fae467"). Pass lines in the file header of
// Sources/UEMAScene/SkinLayout.swift. Nothing here was compiled on the machine that wrote it.
//
// The expected values below were read from the shipped file
// Sources/UEMAScene/Resources/skin_layout_age24_trunk_v1.json on November 4, 2026 (its points
// rows and bodies rows); the counts 4,683 and the digest are the released layout's
// (mimo_bridge/touch_inside_rays.py:24-25). The synthetic document's digest db4fbc87885a0098
// was computed on the same day with Python's hashlib.sha256 over struct.pack("<3d", ...) of
// its three points, the rule of touch_inside_rays.py:126-131.

import Foundation
import XCTest
import UEMAScene

final class SkinLayoutTests: XCTestCase {
    /// The shipped JSON is found and parses to 4,683 points, 53 bodies and 5,005 vertices.
    func testShippedLayoutParsesFourThousandSixHundredEightyThreePoints() throws {
        let layout = try SkinLayout.loadShipped()
        XCTAssertEqual(layout.format, "uema.skin_layout.v1")
        XCTAssertEqual(layout.pointCount, 4683)
        XCTAssertEqual(layout.points.count, 4683)
        XCTAssertEqual(layout.pointPositions.count, 4683)
        XCTAssertEqual(layout.pointBodyIDs.count, 4683)
        XCTAssertEqual(layout.bodies.count, 53)
        XCTAssertEqual(layout.bodyCount, 53)
        XCTAssertEqual(layout.vertexCount, 5005)
        XCTAssertEqual(layout.bodies.reduce(0) { $0 + $1.submeshes.count }, 73)
        XCTAssertEqual(layout.layoutDigest, "0581069917fae467")
        XCTAssertEqual(layout.world.scene, "crib")
        XCTAssertEqual(layout.world.ageMonths, 24.0)
        XCTAssertEqual(layout.world.bodyLayout, "trunk_v1")
        XCTAssertEqual(layout.world.touchInsideRays, "fixed_v1")
        XCTAssertEqual(layout.world.insideTests, 58)
        XCTAssertEqual(layout.world.shapePairs, 58)
        XCTAssertEqual(layout.search.firstSearchMargin, 1.5)
    }

    /// The digest recomputed from the decoded doubles equals the file's and the shipped value.
    func testComputedDigestEqualsTheShippedDigest() throws {
        let layout = try SkinLayout.loadShipped()
        XCTAssertEqual(layout.computedDigest(), layout.layoutDigest)
        XCTAssertEqual(layout.computedDigest(), SkinLayout.shippedDigest)
        XCTAssertNoThrow(try layout.checkShippedIdentity())
    }

    /// The invariants of apple/scripts/tests/test_export_skin_layout.py hold on the shipped file.
    func testShippedLayoutInvariantsHold() throws {
        let layout = try SkinLayout.loadShipped()
        XCTAssertNoThrow(try layout.validate())
    }

    /// The first and last points equal the file's rows exactly (no rounding on the way in).
    func testFirstAndLastPointsEqualTheFileRows() throws {
        let layout = try SkinLayout.loadShipped()
        let first = try XCTUnwrap(layout.points.first)
        XCTAssertEqual(first.bodyID, 2)
        XCTAssertEqual(first.submesh, 0)
        XCTAssertEqual(first.vertex, 0)
        XCTAssertEqual(first.position, SIMD3<Double>(0.04727440776520818, 0.04414165695633231, 0.005132750808875849))
        let last = try XCTUnwrap(layout.points.last)
        XCTAssertEqual(last.bodyID, 56)
        XCTAssertEqual(last.submesh, 1)
        XCTAssertEqual(last.vertex, 22)
        XCTAssertEqual(last.position, SIMD3<Double>(0.021144452592274432, 0.00827345559277654, -4.4007378294519044e-17))
        XCTAssertEqual(layout.pointPositions[0], first.position)
        XCTAssertEqual(layout.pointBodyIDs[4682], 56)
    }

    /// Three body rows equal the file's: hip (the first), upper_body (three submeshes with
    /// inactive vertices) and left_big_toe (the last).
    func testBodyRowsEqualTheFile() throws {
        let layout = try SkinLayout.loadShipped()

        let hip = try XCTUnwrap(layout.body(id: 2))
        XCTAssertEqual(hip.bodyName, "hip")
        XCTAssertEqual(hip.poolerRow, 0)
        XCTAssertEqual(hip.scaleM, 0.025)
        XCTAssertEqual(hip.forceSpreadScaleM, 0.025)
        XCTAssertEqual(hip.sensorCount, 119)
        XCTAssertEqual(hip.pointOffset, 0)
        XCTAssertEqual(hip.submeshes.count, 1)
        let lowerBack = try XCTUnwrap(hip.submeshes.first)
        XCTAssertEqual(lowerBack.geomID, 10)
        XCTAssertEqual(lowerBack.geomName, "lb")
        XCTAssertEqual(lowerBack.geomType, "mjGEOM_CAPSULE")
        XCTAssertEqual(lowerBack.vertexCount, 119)
        XCTAssertEqual(lowerBack.activeCount, 119)
        XCTAssertFalse(lowerBack.isPointCloud)
        XCTAssertEqual(hip.livePositions.count, 119)
        XCTAssertEqual(hip.livePositions[0], layout.pointPositions[0])

        let upperBody = try XCTUnwrap(layout.body(id: 4))
        XCTAssertEqual(upperBody.bodyName, "upper_body")
        XCTAssertEqual(upperBody.pointOffset, 235)
        XCTAssertEqual(upperBody.sensorCount, 136)
        XCTAssertEqual(upperBody.submeshes.map { $0.vertexCount }, [82, 76, 10])
        XCTAssertEqual(upperBody.submeshes.map { $0.activeCount }, [66, 60, 10])
        XCTAssertEqual(upperBody.livePositions.count, 136)
        XCTAssertEqual(upperBody.livePositions, Array(layout.pointPositions[235..<371]))

        let bigToe = try XCTUnwrap(layout.body(id: 56))
        XCTAssertEqual(bigToe.bodyName, "left_big_toe")
        XCTAssertEqual(bigToe.poolerRow, 52)
        XCTAssertEqual(bigToe.scaleM, 0.01)
        XCTAssertEqual(bigToe.pointOffset, 4648)
        XCTAssertEqual(bigToe.sensorCount, 35)
        XCTAssertEqual(bigToe.submeshes.map { $0.geomType }, ["mjGEOM_BOX", "mjGEOM_CYLINDER"])
        XCTAssertEqual(bigToe.submeshes.map { $0.vertexCount }, [18, 23])
        XCTAssertEqual(bigToe.submeshes.map { $0.activeCount }, [16, 19])

        XCTAssertNil(layout.body(id: 45), "left_eye is a placeholder, not a skin body")
        XCTAssertNil(layout.body(id: 0), "the world body has no skin")
        XCTAssertEqual(layout.bodies.map { $0.bodyID }, layout.bodies.map { $0.bodyID }.sorted())
    }

    /// Every neighbour list is within its submesh and no vertex lists itself; the largest list
    /// in the shipped file holds 7 entries (counted on November 4, 2026).
    func testNeighbourListsStayWithinTheirSubmesh() throws {
        let layout = try SkinLayout.loadShipped()
        var largest = 0
        for body in layout.bodies {
            for submesh in body.submeshes {
                for (vertex, list) in submesh.neighbours.enumerated() {
                    largest = max(largest, list.count)
                    XCTAssertFalse(list.contains(vertex))
                    XCTAssertTrue(list.allSatisfy { $0 >= 0 && $0 < submesh.vertexCount })
                }
            }
        }
        XCTAssertEqual(largest, 7)
    }

    /// The tactile reader takes the table through the SkinLayoutReading protocol (SOLID:
    /// dependency inversion); the shipped layout satisfies every requirement.
    func testShippedLayoutServesTheSkinLayoutReadingProtocol() throws {
        let table: SkinLayoutReading = try SkinLayout.loadShipped()
        XCTAssertEqual(table.pointCount, 4683)
        XCTAssertEqual(table.bodies.count, 53)
        XCTAssertEqual(table.pointPositions.count, table.pointBodyIDs.count)
        XCTAssertEqual(table.layoutDigest, SkinLayout.shippedDigest)
        XCTAssertEqual(table.body(id: 2)?.bodyName, "hip")
        XCTAssertNil(table.body(id: 46))
    }

    /// The placeholder bodies are the two eyes, one point each at the origin, spacing 1.0 m.
    func testPlaceholderBodiesAreTheTwoEyes() throws {
        let layout = try SkinLayout.loadShipped()
        XCTAssertEqual(layout.placeholderBodies.map { $0.bodyID }, [45, 46])
        XCTAssertEqual(layout.placeholderBodies.map { $0.bodyName }, ["left_eye", "right_eye"])
        for eye in layout.placeholderBodies {
            XCTAssertEqual(eye.scaleM, 1.0)
            XCTAssertEqual(eye.sensorCount, 1)
            XCTAssertEqual(eye.positionsM, [SIMD3<Double>(0, 0, 0)])
        }
    }

    /// A synthetic two-body document decodes, validates and recomputes its hand-computed digest.
    func testSyntheticLayoutDecodesValidatesAndDigests() throws {
        let layout = try SkinLayout.decode(Data(syntheticJSON.utf8))
        XCTAssertNoThrow(try layout.validate())
        XCTAssertEqual(layout.points.count, 3)
        XCTAssertEqual(layout.computedDigest(), "db4fbc87885a0098")
        XCTAssertEqual(layout.computedDigest(), layout.layoutDigest)
        XCTAssertEqual(layout.pointBodyIDs, [4, 4, 9])
        XCTAssertEqual(layout.pointPositions[2], SIMD3<Double>(1.5, -2.5, 0.0))
        let a = try XCTUnwrap(layout.body(id: 4))
        XCTAssertEqual(a.livePositions, [SIMD3<Double>(0.1, 0.2, 0.3), SIMD3<Double>(0.7, 0.8, 0.9)])
        XCTAssertEqual(a.submeshes[0].sensorIndex, [0, -1, 1])
        XCTAssertEqual(layout.placeholderBodies.count, 1)
        XCTAssertThrowsError(try layout.checkShippedIdentity()) { error in
            guard let layoutError = error as? SkinLayoutError, case .identity = layoutError else {
                return XCTFail("expected identity, got \(error)")
            }
        }
    }

    /// A points row of 5 entries and a geom size of 2 numbers are refused at decode as malformed.
    func testMalformedRowsAreRefusedAtDecode() {
        let shortRow = syntheticJSON.replacingOccurrences(of: "[9,0,0,1.5,-2.5,0.0]", with: "[9,0,0,1.5,-2.5]")
        XCTAssertNotEqual(shortRow, syntheticJSON)
        XCTAssertThrowsError(try SkinLayout.decode(Data(shortRow.utf8))) { error in
            guard let layoutError = error as? SkinLayoutError, case let .malformed(text) = layoutError else {
                return XCTFail("expected malformed, got \(error)")
            }
            XCTAssertTrue(text.contains("5 entries"), text)
        }
        let shortSize = syntheticJSON.replacingOccurrences(of: "\"geom_size_m\":[0.05,0.0,0.0]", with: "\"geom_size_m\":[0.05,0.0]")
        XCTAssertNotEqual(shortSize, syntheticJSON)
        XCTAssertThrowsError(try SkinLayout.decode(Data(shortSize.utf8))) { error in
            guard let layoutError = error as? SkinLayoutError, case let .malformed(text) = layoutError else {
                return XCTFail("expected malformed, got \(error)")
            }
            XCTAssertTrue(text.contains("geom_size_m"), text)
        }
        XCTAssertThrowsError(try SkinLayout.decode(Data("{\"format\": 1}".utf8)))
    }

    /// A wrong point offset, a wrong sensor index, a one-sided edge, two points rows out of
    /// digest order and a wrong format are refused by validate() with the body, row or value
    /// named; an absent file is refused as unreadable.
    func testBrokenInvariantsAreRefusedByName() throws {
        let wrongOffset = syntheticJSON.replacingOccurrences(of: "\"point_offset\":2", with: "\"point_offset\":1")
        XCTAssertNotEqual(wrongOffset, syntheticJSON)
        XCTAssertThrowsError(try SkinLayout.decode(Data(wrongOffset.utf8)).validate()) { error in
            guard let layoutError = error as? SkinLayoutError, case let .invariant(text) = layoutError else {
                return XCTFail("expected invariant, got \(error)")
            }
            XCTAssertTrue(text.contains("body 9 point_offset 1"), text)
        }
        let wrongIndex = syntheticJSON.replacingOccurrences(of: "\"sensor_index\":[0,-1,1]", with: "\"sensor_index\":[0,-1,2]")
        XCTAssertNotEqual(wrongIndex, syntheticJSON)
        XCTAssertThrowsError(try SkinLayout.decode(Data(wrongIndex.utf8)).validate()) { error in
            guard let layoutError = error as? SkinLayoutError, case let .invariant(text) = layoutError else {
                return XCTFail("expected invariant, got \(error)")
            }
            XCTAssertTrue(text.contains("body 4 submesh 0 vertex 2 sensor_index 2"), text)
        }
        let oneSidedEdge = syntheticJSON.replacingOccurrences(of: "\"neighbours\":[[1],[0,2],[1]]", with: "\"neighbours\":[[1],[0,2],[]]")
        XCTAssertNotEqual(oneSidedEdge, syntheticJSON)
        XCTAssertThrowsError(try SkinLayout.decode(Data(oneSidedEdge.utf8)).validate()) { error in
            guard let layoutError = error as? SkinLayoutError, case let .invariant(text) = layoutError else {
                return XCTFail("expected invariant, got \(error)")
            }
            XCTAssertTrue(text.contains("not symmetric"), text)
        }
        let swappedRows = syntheticJSON.replacingOccurrences(
            of: "[[4,0,0,0.1,0.2,0.3],[4,0,2,0.7,0.8,0.9]",
            with: "[[4,0,2,0.7,0.8,0.9],[4,0,0,0.1,0.2,0.3]")
        XCTAssertNotEqual(swappedRows, syntheticJSON)
        XCTAssertThrowsError(try SkinLayout.decode(Data(swappedRows.utf8)).validate()) { error in
            guard let layoutError = error as? SkinLayoutError, case let .invariant(text) = layoutError else {
                return XCTFail("expected invariant, got \(error)")
            }
            XCTAssertTrue(text.contains("points row 0 is [4, 0, 2], expected [4, 0, 0]"), text)
        }
        let wrongFormat = syntheticJSON.replacingOccurrences(of: "uema.skin_layout.v1", with: "uema.skin_layout.v0")
        XCTAssertNotEqual(wrongFormat, syntheticJSON)
        XCTAssertThrowsError(try SkinLayout.decode(Data(wrongFormat.utf8)).validate()) { error in
            guard let layoutError = error as? SkinLayoutError, case let .invariant(text) = layoutError else {
                return XCTFail("expected invariant, got \(error)")
            }
            XCTAssertTrue(text.contains("uema.skin_layout.v0"), text)
        }
        let missingFile = FileManager.default.temporaryDirectory.appendingPathComponent("no-such-\(UUID().uuidString).json")
        XCTAssertThrowsError(try SkinLayout.load(from: missingFile)) { error in
            guard let layoutError = error as? SkinLayoutError, case .unreadable = layoutError else {
                return XCTFail("expected unreadable, got \(error)")
            }
        }
    }

    /// A synthetic document in the shipped format: body 4 with one box mesh of 3 vertices (the
    /// middle one inactive) and body 9 with one single-point mesh; the two eyes reduced to one
    /// placeholder. Its three live points are (0.1, 0.2, 0.3), (0.7, 0.8, 0.9), (1.5, -2.5, 0.0).
    private let syntheticJSON = """
    {"format":"uema.skin_layout.v1","written_by":"SkinLayoutTests","written_at":"test",
    "world":{"scene":"crib","age_months":24.0,"body_layout":"trunk_v1","contact_profile":"mechanical_v4","touch_inside_rays":"fixed_v1","touch_scale_mult":1.0,"directions_sha256":"none","inside_tests":1,"shape_pairs":1,"nq":1,"nbody":2,"scene_xml":"s.xml","scene_additions":""},
    "search":{"scale":"s","distance_limit":"d","first_search_margin":1.5,"weight":"w","source":"src"},
    "layout_digest":"db4fbc87885a0098","point_count":3,"vertex_count":4,"body_count":2,
    "bodies":[
    {"body_id":4,"body_name":"a","pooler_row":0,"scale_m":0.01,"force_spread_scale_m":0.01,"sensor_count":2,"point_offset":0,
    "submeshes":[{"submesh":0,"geom_id":7,"geom_name":"ga","geom_type":"mjGEOM_BOX","geom_size_m":[0.1,0.2,0.3],"is_point_cloud":false,"vertex_count":3,"active_count":2,
    "positions_m":[[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]],"active":[true,false,true],"sensor_index":[0,-1,1],"neighbours":[[1],[0,2],[1]]}]},
    {"body_id":9,"body_name":"b","pooler_row":1,"scale_m":0.02,"force_spread_scale_m":0.02,"sensor_count":1,"point_offset":2,
    "submeshes":[{"submesh":0,"geom_id":8,"geom_name":"gb","geom_type":"mjGEOM_SPHERE","geom_size_m":[0.05,0.0,0.0],"is_point_cloud":true,"vertex_count":1,"active_count":1,
    "positions_m":[[1.5,-2.5,0.0]],"active":[true],"sensor_index":[0],"neighbours":[[]]}]}],
    "points":[[4,0,0,0.1,0.2,0.3],[4,0,2,0.7,0.8,0.9],[9,0,0,1.5,-2.5,0.0]],
    "placeholder_bodies":{"rule":"r","bodies":[{"body_id":45,"body_name":"left_eye","scale_m":1.0,"sensor_count":1,"positions_m":[[0.0,0.0,0.0]]}]}}
    """
}
