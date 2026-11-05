// TactileFixtureTests.swift: the tactile stages against the Python fixture of work item WI-14
// (Fixtures/eyes/<state>/tactile_patches.json, written by apple/scripts/export_eye_fixtures.py,
// format "uema.eye_fixture.tactile_patches.v1"), work item WI-28's acceptance line "WI-14's
// tactile_patches.json matched to a tolerance measured at first build and written into
// section 9".
//
// What each fixture holds, per state (step_000492800, step_000493674, step_000493980 of the
// released run B1K_skip13_492800): the 53 skin bodies (`patches`: body id, name, spacing,
// point count); `inputs.contacts` (4 to 8 records of env.py `_contacts`, qvel zero);
// `inputs.point_forces_body_frame_n` (MIMo `get_touch_obs` with the bridge's
// `conservative_spread_linear`, every point with a non-zero force: 57 to 90 points); `fields`
// [53][10], the result of `TouchPooler.update` on a fresh pooler (dt 0.01 s, window 100).
//
// The three checks per state, each printing "WI-28 measured difference <name> <state>: <value>":
//  1. spread: the fixture's contacts (force on the body = normal force x body-frame normal +
//     body-frame tangential force, the MIMo `force_vector` of touch.py:1490-1504) through
//     `LinearContactSpreader` over the shipped skin table, against the fixture's point forces;
//     largest absolute difference over every point and component, N. Provisional line 1e-5 N
//     (float32 resolution at 2 N is 2.4e-7 N; the line is replaced by the measured value).
//     MIMo's `get_contact_normal` (touch.py:1455-1458: `normal_vector = contact.frame[:3]`,
//     then `normal_vector *= -1` when geom 1 belongs to the body) negates MuJoCo's contact
//     frame in place. The exporter ran `get_touch_obs` and then `env._contacts` on the same
//     state (export_eye_fixtures.py:381-383), so for a contact whose geom-1 body has skin and
//     whose dist is negative the recorded records differ from what MIMo's spread used: the
//     geom-1-side record carries the negated normal, and both records' `pos_body` sit
//     |dist| = penetration away from the positions MIMo spread at, along the recorded
//     body-frame normal. `mimoSpreadContacts` undoes exactly that (true normal = -recorded on
//     the geom-1 side; spread position = recorded + recorded normal x penetration on both
//     sides). The same rule run in numpy on the three fixtures (November 4, 2026) reproduces
//     every recorded point force with difference 0; without it the self-touch bodies differ
//     by up to 0.49 N (left_thhub, step_000493674) and every other body by 0.
//     `ContactReader` computes the physical normal for both sides and does not reproduce the
//     in-place negation.
//  2. pooler: the fixture's point forces and contacts through `GoertzelTouchPooler`, against
//     `fields`; largest absolute difference after dividing both by `_FIELD_NORMS`
//     (dimensionless; REQ-B-5's line 1e-4 per normalised field).
//  3. table: the shipped table's recomputed digest equals 0581069917fae467 and its 53 bodies'
//     ids and spacings equal the fixture's `patches`; UEMAScene's `SkinLayout` wrapped by
//     `SkinLayoutTable` (SkinLayoutAdapter.swift) gives the same bodies, spread scales and
//     live positions as the JSON decoded here.
// Nothing in this file was compiled on the machine that wrote it (no Swift toolchain).

import Foundation
import XCTest
import UEMAScene
import UEMASenses

/// The parts of tactile_patches.json these tests read.
private struct TactileFixture: Decodable {
    struct Patch: Decodable {
        let row: Int
        let bodyId: Int
        let bodyName: String
        let scale: Double
        let sensorCount: Int
        enum CodingKeys: String, CodingKey {
            case row
            case bodyId = "body_id"
            case bodyName = "body_name"
            case scale = "scale_m"
            case sensorCount = "sensor_count"
        }
    }

    struct FixtureContact: Decodable {
        let bodyId: Int
        let otherBodyName: String
        let pairKey: [Int]
        let normalForce: Double
        let tangentialBody: [Double]
        let positionBody: [Double]
        let positionWorld: [Double]
        let normalWorld: [Double]
        let normalBody: [Double]
        let penetration: Double
        let closingSpeed: Double
        enum CodingKeys: String, CodingKey {
            case bodyId = "body_id"
            case otherBodyName = "other_body_name"
            case pairKey = "pair_key"
            case normalForce = "normal_force_n"
            case tangentialBody = "tangential_body_n"
            case positionBody = "pos_body_m"
            case positionWorld = "pos_world_m"
            case normalWorld = "normal_world"
            case normalBody = "normal_body"
            case penetration = "penetration_m"
            case closingSpeed = "rel_normal_speed_m_s"
        }
    }

    struct Inputs: Decodable {
        let pointForces: [String: [[Double]]]
        let contacts: [FixtureContact]
        enum CodingKeys: String, CodingKey {
            case pointForces = "point_forces_body_frame_n"
            case contacts
        }
    }

    let format: String
    let dt: Double
    let vibrationWindow: Int
    let patchCount: Int
    let fields: [[Double]]
    let patches: [Patch]
    let inputs: Inputs

    enum CodingKeys: String, CodingKey {
        case format
        case dt = "dt_s"
        case vibrationWindow = "vib_window"
        case patchCount = "n_patches"
        case fields
        case patches
        case inputs
    }
}

final class TactileFixtureTests: XCTestCase {
    /// The three exported states (Fixtures/eyes/).
    private let states = ["step_000492800", "step_000493674", "step_000493980"]
    /// Provisional line for the spread, N (file header, check 1).
    private let spreadLine = 1e-5
    /// REQ-B-5's line per normalised field.
    private let normalisedLine = 1e-4

    private func fixture(_ state: String) throws -> TactileFixture {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "tactile_patches", withExtension: "json",
                              subdirectory: "Fixtures/eyes/\(state)"),
            "Fixtures/eyes/\(state)/tactile_patches.json is not in the test bundle")
        let fixture = try JSONDecoder().decode(TactileFixture.self, from: Data(contentsOf: url))
        XCTAssertEqual(fixture.format, "uema.eye_fixture.tactile_patches.v1")
        return fixture
    }

    private func shippedTable() throws -> ShippedSkinTable {
        try ShippedSkinTable(contentsOf: SkinLayout.shippedURL())
    }

    private func vec(_ a: [Double]) throws -> Vec3 {
        try XCTUnwrap(Vec3(array: a), "a vector of \(a.count) numbers")
    }

    private func contacts(_ f: TactileFixture) throws -> [SkinContact] {
        try f.inputs.contacts.map { c in
            let normalBody = try vec(c.normalBody)
            let tangentialBody = try vec(c.tangentialBody)
            return SkinContact(
                bodyId: c.bodyId, geom1: c.pairKey[0], geom2: c.pairKey[1],
                normalForce: c.normalForce, tangentialBody: tangentialBody,
                positionBody: try vec(c.positionBody), positionWorld: try vec(c.positionWorld),
                normalWorld: try vec(c.normalWorld), normalBody: normalBody,
                forceBody: normalBody * c.normalForce + tangentialBody,
                penetration: c.penetration, closingSpeed: c.closingSpeed,
                tangentSpeed: nil, frictionPower: nil,
                otherBodyName: c.otherBodyName, otherIsSelf: false)
        }
    }

    /// The contacts as MIMo's spread saw them (file header, check 1): for a geom pair whose
    /// geom 1 belongs to a skin body (`submeshes[].geom_id`) and whose penetration is above
    /// 0, the geom-1-side normal is negated back and both sides' positions move by
    /// recorded normal x penetration.
    private func mimoSpreadContacts(_ f: TactileFixture, table: ShippedSkinTable) throws -> [SkinContact] {
        func geomOneSide(_ c: TactileFixture.FixtureContact) -> Bool {
            guard let body = table.body(id: c.bodyId) else { return false }
            return body.submeshes.contains { $0.geomId == c.pairKey[0] }
        }
        var flippedPairs = Set<GeomPair>()
        for c in f.inputs.contacts where geomOneSide(c) && c.penetration > 0 {
            flippedPairs.insert(GeomPair(geom1: c.pairKey[0], geom2: c.pairKey[1]))
        }
        return try f.inputs.contacts.map { c in
            let recordedNormal = try vec(c.normalBody)
            let tangentialBody = try vec(c.tangentialBody)
            var normalBody = recordedNormal
            var positionBody = try vec(c.positionBody)
            if flippedPairs.contains(GeomPair(geom1: c.pairKey[0], geom2: c.pairKey[1])) {
                if geomOneSide(c) { normalBody = -recordedNormal }
                positionBody = positionBody + recordedNormal * c.penetration
            }
            return SkinContact(
                bodyId: c.bodyId, geom1: c.pairKey[0], geom2: c.pairKey[1],
                normalForce: c.normalForce, tangentialBody: tangentialBody,
                positionBody: positionBody, positionWorld: try vec(c.positionWorld),
                normalWorld: try vec(c.normalWorld), normalBody: normalBody,
                forceBody: normalBody * c.normalForce + tangentialBody,
                penetration: c.penetration, closingSpeed: c.closingSpeed,
                tangentSpeed: nil, frictionPower: nil,
                otherBodyName: c.otherBodyName, otherIsSelf: false)
        }
    }

    /// The fixture's point forces as a full table (unlisted points zero).
    private func recordedForces(_ f: TactileFixture, table: ShippedSkinTable) throws -> SkinPointForces {
        var out = SkinPointForces(table: table)
        for (key, rows) in f.inputs.pointForces {
            let body = try XCTUnwrap(Int(key), "body key \(key)")
            for row in rows {
                XCTAssertEqual(row.count, 4)
                let ok = out.set(Vec3(row[1], row[2], row[3]), body: body, point: Int(row[0]))
                XCTAssertTrue(ok, "body \(body) point \(Int(row[0])) is not in the shipped table")
            }
        }
        return out
    }

    /// Check 3: the table the other checks run over.
    func testShippedTableMatchesTheFixturePatches() throws {
        let table = try shippedTable()
        XCTAssertEqual(table.pointCount, 4683)
        XCTAssertEqual(table.layoutDigest, "0581069917fae467")
        XCTAssertEqual(table.recomputedDigest(), "0581069917fae467")
        let f = try fixture(states[0])
        XCTAssertEqual(f.patchCount, 53)
        let specs = table.patchSpecs.filter { $0.scale < GoertzelTouchPooler.skinScaleMax }
        XCTAssertEqual(specs.map { $0.bodyId }, f.patches.map { $0.bodyId })
        XCTAssertEqual(specs.map { $0.pointCount }, f.patches.map { $0.sensorCount })
        for (s, p) in zip(specs, f.patches) {
            XCTAssertEqual(s.scale, p.scale, accuracy: 1e-12, "body \(p.bodyId) \(p.bodyName)")
        }
    }

    /// Check 3, second half: the adapter over UEMAScene's layout equals the decoded table.
    func testSkinLayoutAdapterMatchesTheDecodedTable() throws {
        let decoded = try shippedTable()
        let adapted = try SkinLayoutTable.shipped()
        XCTAssertEqual(adapted.pointCount, decoded.pointCount)
        XCTAssertEqual(adapted.layoutDigest, decoded.layoutDigest)
        XCTAssertEqual(adapted.bodies.map { $0.bodyId }, decoded.bodies.map { $0.bodyId })
        for (a, d) in zip(adapted.bodies, decoded.bodies) {
            XCTAssertEqual(a.forceSpreadScale, d.forceSpreadScale, "body \(d.bodyId) \(d.bodyName)")
            XCTAssertEqual(a.scale, d.scale, "body \(d.bodyId) \(d.bodyName)")
            XCTAssertEqual(a.livePositions, d.livePositions, "body \(d.bodyId) \(d.bodyName)")
            XCTAssertEqual(a.submeshes.map { $0.neighbours }, d.submeshes.map { $0.neighbours })
        }
    }

    /// Check 1: the spread of the fixture's contacts against MIMo's point forces.
    func testSpreadMatchesTheFixturePointForces() throws {
        let table = try shippedTable()
        let spreader = LinearContactSpreader(table: table)
        for state in states {
            let f = try fixture(state)
            let expected = try recordedForces(f, table: table)
            let got = spreader.pointForces(for: try mimoSpreadContacts(f, table: table), table: table)
            var largest = 0.0
            var worst = ""
            for body in table.bodies {
                let e = expected.forces(ofBody: body.bodyId)
                let g = got.forces(ofBody: body.bodyId)
                XCTAssertEqual(e.count, g.count)
                for i in 0 ..< min(e.count, g.count) {
                    let d = max(abs(e[i].x - g[i].x), abs(e[i].y - g[i].y), abs(e[i].z - g[i].z))
                    if d > largest {
                        largest = d
                        worst = "body \(body.bodyId) point \(i): fixture \(e[i].array), swift \(g[i].array)"
                    }
                }
            }
            print("WI-28 measured difference spread_point_force_N \(state): \(largest) (\(worst))")
            XCTAssertLessThanOrEqual(largest, spreadLine, "\(state): \(worst)")
        }
    }

    /// Check 2: the pooler on the fixture's own inputs against its fields.
    func testPoolerMatchesTheFixtureFields() throws {
        let table = try shippedTable()
        for state in states {
            let f = try fixture(state)
            let pooler = try GoertzelTouchPooler(specs: table.patchSpecs, dt: f.dt, vibrationWindow: f.vibrationWindow)
            XCTAssertEqual(pooler.patchCount, f.patchCount)
            let out = pooler.update(pointForces: try recordedForces(f, table: table),
                                    contacts: try contacts(f), includeVibration: true)
            var largestNormalised = 0.0
            var largestRaw = 0.0
            var worst = ""
            for p in 0 ..< min(out.patchCount, f.fields.count) {
                for c in 0 ..< 10 {
                    let raw = abs(Double(out.fields[p][c]) - f.fields[p][c])
                    let normalised = raw / Double(TactileFieldNorms.divisors[c])
                    largestRaw = max(largestRaw, raw)
                    if normalised > largestNormalised {
                        largestNormalised = normalised
                        worst = "patch \(p) field \(c): fixture \(f.fields[p][c]), swift \(out.fields[p][c])"
                    }
                }
            }
            print("WI-28 measured difference pooler_normalised_field \(state): \(largestNormalised) (\(worst))")
            print("WI-28 measured difference pooler_raw_field \(state): \(largestRaw)")
            XCTAssertLessThanOrEqual(largestNormalised, normalisedLine, "\(state): \(worst)")
        }
    }

    /// The token of the fixture's fields: every active row has its valid flag and the rows are
    /// in descending normalised pressure (lem_model.py:231-267); positions are not in the
    /// fixture, so columns 0-2 read 0.
    func testFixtureFieldsGiveAPressureSortedToken() throws {
        for state in states {
            let f = try fixture(state)
            let slots = try TactileFieldNorms.normalise(f.fields.map { $0.map { Float($0) } })
            let token = PressureSortedPointCloud().build(normalisedSlots: slots, patchPositions: nil,
                                                         selfColumns: nil, contactColumns: nil)
            let active = slots.filter { row in row.reduce(Float(0)) { $0 + abs($1) } > 0 }.count
            XCTAssertEqual(token.activeRowCount, min(active, 32), state)
            let pressures = token.rows.prefix(token.activeRowCount).map { $0[3] }
            XCTAssertEqual(pressures, pressures.sorted(by: >), state)
        }
    }
}
