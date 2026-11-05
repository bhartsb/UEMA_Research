// TactileSenseTests.swift: hand-computed checks of the contact reader (ContactReader.swift),
// the physics view (PhysicsContactView.swift), the point-cloud token (TouchPooler.swift) and
// the session-facing `TactileSense` (TactileSense.swift), work item WI-28. Acceptance line
// "0 contacts give 32 zero rows" is `testNoContactsGiveThirtyTwoZeroRows`.
//
// Worked contact (testOneFloorContactInTheBodyFrame): the floor (body 0, geom 0) under skin
// body 5 (geom 3), contact frame rows (0, 0, 1), (1, 0, 0), (0, 1, 0) (normal +z, from geom 0
// into geom 3),
// mj_contactForce (10, 2, 0) N, midpoint (1, 0, 0.1) m, dist -0.002 m; body 5 at (1, 0, 0) m
// turned 90 degrees about z (xmat rows (0, -1, 0), (1, 0, 0), (0, 0, 1)), moving at
// (0.1, 0, -0.3) m/s.
//   f_world = 10 (0, 0, 1) + 2 (1, 0, 0) = (2, 0, 10) N; body 5 is geom 2's body: n_into (0, 0, 1)
//   tang_world = (2, 0, 0) N; R^T (2, 0, 0) = (0, -2, 0) N; force_body = R^T (2, 0, 10) = (0, -2, 10) N
//   pos_body = R^T (0, 0, 0.1) + (0, 0, 1) x (-0.002 / 2) = (0, 0, 0.099) m
//   rel = v_floor - v_body = (-0.1, 0, 0.3) m/s; closing 0.3 m/s; tangent speed 0.1 m/s;
//   friction power max(0, (2, 0, 0) . (-0.1, 0, 0.3)) = max(0, -0.2) = 0 W
// Worked token (testOneContactThroughTheSense): one 10 N contact at the origin of a three-point
// body (spacing 0.01 m, points at x = 0, 0.01, 0.015 m): all three points carry force, area
// 3 x 100 mm2 = 300 mm2, pressure 10 N / 0.0003 m2 = 33,333.3 Pa, over 100,000 Pa = 0.33333;
// area 300 / 20,000 = 0.015; the vibration bands are the Goertzel power of a 100-sample history
// ending in five samples of 10 N (one per env step), over 10 N^2; patch body 0.5 m above the
// root body: 0.5 / 0.5 = 1.0.
// Worked self-touch (testFrameNegationRules): skin body 5 (geom 3, geom 1 of the contact) on
// skin body 6 (geom 4), normal +z from geom 3 into geom 4, dist -0.002 m, fn 1 N, both bodies
// at the origin unrotated, body 5 moving at (0, 0, 0.1) m/s, body 6 still.
//   physical: body 5 reads n_into (0, 0, -1), pos_body (0, 0, -1) x (-0.002 / 2) = (0, 0, 0.001) m,
//             rel = -(v5 - v6) = (0, 0, -0.1) m/s, closing 0.1 m/s; body 6 reads (0, 0, 1),
//             pos_body (0, 0, -0.001) m, rel (0, 0, 0.1) m/s, closing 0.1 m/s.
//   geom2Side: body 6 reads (0, 0, -1), pos_body (0, 0, 0.001) m, closing max(0, -0.1) = 0.
//   geom1Side: body 5 reads (0, 0, 1), pos_body (0, 0, -0.001) m, closing max(0, -0.1) = 0.
//   force_body is (0, 0, -1) N on body 5 and (0, 0, 1) N on body 6 under every rule.
// Nothing in this file was compiled on the machine that wrote it (no Swift toolchain).

import Foundation
import XCTest
import UEMACore
import UEMASenses

/// Hand-written arrays answering `ContactPhysicsReading`.
private struct TactileFakeContactPhysics: ContactPhysicsReading {
    var contacts: [RawContact] = []
    var forces: [ContactForce] = []
    var geomBodies: [Int] = []
    var roots: [Int] = []
    var names: [String] = []
    var positions: [Int: Vec3] = [:]
    var rotations: [Int: Mat3] = [:]
    var velocities: [Int: Vec3] = [:]

    var contactCount: Int { contacts.count }
    func contact(at index: Int) -> RawContact { contacts[index] }
    func contactForce(at index: Int) -> ContactForce { forces[index] }
    func bodyId(ofGeom geomId: Int) -> Int { geomBodies[geomId] }
    func rootBodyId(ofBody bodyId: Int) -> Int { roots[bodyId] }
    func bodyName(_ bodyId: Int) -> String { names[bodyId] }
    func bodyPosition(_ bodyId: Int) -> Vec3 { positions[bodyId] ?? .zero }
    func bodyRotation(_ bodyId: Int) -> Mat3 { rotations[bodyId] ?? .identity }
    func pointVelocity(ofBody bodyId: Int, atWorldPoint point: Vec3) -> Vec3 { velocities[bodyId] ?? .zero }
}

/// Hand-written arrays answering UEMACore's `PhysicsReading` (8 bodies, 2 geoms).
private final class TactileFakePhysics: PhysicsReading {
    var contactList: [Contact] = []
    var bodyPositions: [Double] = [Double](repeating: 0, count: 8 * 3)
    var velocity6: [Int: [Double]] = [:]

    var sizes: PhysicsSizes { PhysicsSizes(nq: 0, nv: 0, nu: 0, nbody: 8, njnt: 0, nsensordata: 0, ngeom: 2, ncam: 0) }
    var time: Double { 0 }
    var qpos: [Double] { [] }
    var qvel: [Double] { [] }
    var sensordata: [Double] { [] }
    var xpos: [Double] { bodyPositions }
    var xquat: [Double] { [] }
    var xmat: [Double] {
        (0 ..< 8).flatMap { _ in [1.0, 0, 0, 0, 1, 0, 0, 0, 1] }
    }
    var xaxis: [Double] { [] }
    var xanchor: [Double] { [] }
    var actuatorForce: [Double] { [] }
    var control: [Double] { [] }
    var contacts: [Contact] { contactList }
    func id(of kind: MuJoCoObjectKind, named name: String) -> Int? { nil }
    func bodyVelocity(bodyId: Int) -> [Double] { velocity6[bodyId] ?? [0, 0, 0, 0, 0, 0] }
}

/// A table of hand-built bodies (the `SkinTable` of UEMASenses).
private struct TactileHandTable: SkinTable {
    let bodies: [SkinBodyRows]
    func body(id: Int) -> SkinBodyRows? { bodies.first { $0.bodyId == id } }
    var pointCount: Int { bodies.reduce(0) { $0 + $1.sensorCount } }
    var layoutDigest: String { "hand" }
}

final class TactileSenseTests: XCTestCase {
    /// Contact frame rows: normal (0, 0, 1), tangent 1 (1, 0, 0), tangent 2 (0, 1, 0).
    private let zFrame = Mat3(rowMajor: [0, 0, 1, 1, 0, 0, 0, 1, 0])!

    private func assertVector(_ v: Vec3, _ x: Double, _ y: Double, _ z: Double, accuracy: Double = 1e-12,
                              file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(v.x, x, accuracy: accuracy, file: file, line: line)
        XCTAssertEqual(v.y, y, accuracy: accuracy, file: file, line: line)
        XCTAssertEqual(v.z, z, accuracy: accuracy, file: file, line: line)
    }

    /// Bodies 0 world, 5 and 6 skin (root 1); geoms 0 -> body 0, 3 -> body 5, 4 -> body 6.
    private func physics() -> TactileFakeContactPhysics {
        var p = TactileFakeContactPhysics()
        p.geomBodies = [0, 0, 0, 5, 6]
        p.roots = [0, 1, 1, 1, 1, 1, 1]
        p.names = ["world", "hip", "b2", "b3", "b4", "body5", "body6"]
        return p
    }

    /// The worked contact of the file header.
    func testOneFloorContactInTheBodyFrame() throws {
        var p = physics()
        p.contacts = [RawContact(geom1: 0, geom2: 3, position: Vec3(1, 0, 0.1), frame: zFrame, distance: -0.002)]
        p.forces = [ContactForce(normal: 10, tangent1: 2, tangent2: 0)]
        p.positions[5] = Vec3(1, 0, 0)
        p.rotations[5] = try XCTUnwrap(Mat3(rowMajor: [0, -1, 0, 1, 0, 0, 0, 0, 1]))
        p.velocities[5] = Vec3(0.1, 0, -0.3)
        let records = ContactReader(skinBodyIds: [5, 6], ownRootBodyId: 1).read(p)
        XCTAssertEqual(records.count, 1)
        let r = records[0]
        XCTAssertEqual(r.bodyId, 5)
        XCTAssertEqual(r.normalForce, 10, accuracy: 1e-12)
        assertVector(r.normalWorld, 0, 0, 1)
        assertVector(r.tangentialBody, 0, -2, 0)
        assertVector(r.forceBody, 0, -2, 10)
        assertVector(r.normalBody, 0, 0, 1)
        assertVector(r.positionBody, 0, 0, 0.099)
        XCTAssertEqual(r.penetration, 0.002, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(r.closingSpeed), 0.3, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(r.tangentSpeed), 0.1, accuracy: 1e-12)
        XCTAssertEqual(try XCTUnwrap(r.frictionPower), 0.0, accuracy: 0)
        XCTAssertEqual(r.otherBodyName, "world")
        XCTAssertFalse(r.otherIsSelf)
        XCTAssertEqual(r.pairKey, ContactPairKey(bodyId: 5, geom1: 0, geom2: 3))
    }

    /// A self-touch gives two records with the same geom pair, geom 1's body first; on that
    /// side the normal and the force are reversed (env.py:2328-2330).
    func testSelfTouchGivesTwoRecords() {
        var p = physics()
        p.contacts = [RawContact(geom1: 3, geom2: 4, position: .zero, frame: zFrame, distance: 0)]
        p.forces = [ContactForce(normal: 1, tangent1: 0, tangent2: 0)]
        let records = ContactReader(skinBodyIds: [5, 6], ownRootBodyId: 1).read(p)
        XCTAssertEqual(records.map { $0.bodyId }, [5, 6])
        assertVector(records[0].normalWorld, 0, 0, -1)
        assertVector(records[0].forceBody, 0, 0, -1)
        assertVector(records[1].forceBody, 0, 0, 1)
        XCTAssertTrue(records[0].otherIsSelf)
        XCTAssertEqual(records[0].otherBodyName, "body6")
        XCTAssertEqual(records[0].pairKey.geomPair, records[1].pairKey.geomPair)
    }

    /// |fn| under 1e-9 N and a contact between two bodies without skin give no record.
    func testInactiveAndSkinlessContactsAreSkipped() {
        var p = physics()
        p.contacts = [
            RawContact(geom1: 0, geom2: 3, position: .zero, frame: zFrame, distance: 0),
            RawContact(geom1: 0, geom2: 1, position: .zero, frame: zFrame, distance: 0),
        ]
        p.forces = [ContactForce(normal: 1e-10, tangent1: 0, tangent2: 0), ContactForce(normal: 5, tangent1: 0, tangent2: 0)]
        XCTAssertTrue(ContactReader(skinBodyIds: [5, 6], ownRootBodyId: 1).read(p).isEmpty)
    }

    /// The rigid-body velocity: v = (0, 0, 0) + (0, 0, 2) rad/s x ((1, 0, 0) - (0, 0, 0)) m = (0, 2, 0) m/s.
    func testRigidPointVelocity() throws {
        let fake = TactileFakePhysics()
        fake.velocity6[3] = [0, 0, 2, 0, 0, 0]
        let v = RigidPointVelocity.atBodyOrigin.pointVelocity(ofBody: 3, atWorldPoint: Vec3(1, 0, 0), in: fake)
        assertVector(v, 0, 2, 0)
    }

    /// The physics view reads `Contact` values and the tables by index.
    func testPhysicsContactViewReadsTheSharedArrays() throws {
        let fake = TactileFakePhysics()
        fake.contactList = [try Contact(geomIds: [0, 1], distance: -0.001, position: [1, 2, 3],
                                        frame: [1, 0, 0, 0, 1, 0, 0, 0, 1], force: [4, 5, 6, 0, 0, 0])]
        fake.bodyPositions[3 * 7 + 2] = 0.25
        let tables = ContactModelTables(geomBodyIds: [0, 7], bodyRootIds: [0, 1, 1, 1, 1, 1, 1, 1],
                                        bodyNames: ["world", "hip", "b2", "b3", "b4", "b5", "b6", "palm"])
        let view = PhysicsContactView(physics: fake, tables: tables, velocity: RigidPointVelocity.atBodyOrigin)
        XCTAssertEqual(view.contactCount, 1)
        let c = view.contact(at: 0)
        XCTAssertEqual(c.geom2, 1)
        assertVector(c.position, 1, 2, 3)
        XCTAssertEqual(c.distance, -0.001, accuracy: 0)
        XCTAssertEqual(view.contactForce(at: 0).tangent2, 6, accuracy: 0)
        XCTAssertEqual(view.bodyId(ofGeom: 1), 7)
        XCTAssertEqual(view.bodyId(ofGeom: -1), -1)
        XCTAssertEqual(view.bodyName(7), "palm")
        assertVector(view.bodyPosition(7), 0, 0, 0.25)
        XCTAssertEqual(tables.infantRootBodyId, 1)
    }

    /// The three-point body of the worked token (id 7, spacing 0.01 m).
    private func lineTable() throws -> TactileHandTable {
        let mesh = SkinMeshRows(positions: [Vec3(0, 0, 0), Vec3(0.01, 0, 0), Vec3(0.015, 0, 0)],
                                active: [true, true, true], sensorIndex: [0, 1, 2],
                                neighbours: [[1], [0, 2], [1]])
        let body = try XCTUnwrap(SkinBodyRows(bodyId: 7, bodyName: "palm", scale: 0.01, submeshes: [mesh]))
        return TactileHandTable(bodies: [body])
    }

    private func modelTables() -> ContactModelTables {
        ContactModelTables(geomBodyIds: [0, 7], bodyRootIds: [0, 1, 1, 1, 1, 1, 1, 1],
                           bodyNames: ["world", "hip", "b2", "b3", "b4", "b5", "b6", "palm"])
    }

    /// One model step of the sense's cadence: 10 physics-step reads, 5 env-step pools, one read.
    private func runModelStep(_ sense: TactileSense, _ fake: TactileFakePhysics) throws -> SenseReading {
        for _ in 0 ..< ModelStep.envStepsPerModelStep {
            for _ in 0 ..< ModelStep.physicsStepsPerEnvStep {
                try sense.observePhysicsStep(fake)
            }
            try sense.observeEnvStep(fake)
        }
        return try sense.read(fake, at: ModelStep.start)
    }

    /// Acceptance: no contacts give a token of 32 rows x 19 columns, every number 0.
    func testNoContactsGiveThirtyTwoZeroRows() throws {
        let sense = try TactileSense.released(table: lineTable(), tables: modelTables(),
                                              velocity: RigidPointVelocity.atBodyOrigin)
        XCTAssertEqual(sense.sense, .tactile)
        let reading = try runModelStep(sense, TactileFakePhysics())
        XCTAssertTrue(reading.isFresh)
        // one patch row per patch (one patch here), the layout TactileTokenizer reads
        XCTAssertEqual(reading.primary.shape, [1, 10])
        XCTAssertEqual(reading.auxiliary?.shape, [1, 10])
        // the patch row carries the stretch ratios 1.0 and nothing else
        XCTAssertEqual(reading.primary.values, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        let token = try PressureSortedPointCloud().build(reading: reading)
        XCTAssertEqual(token.rows.count, 32)
        XCTAssertTrue(token.rows.allSatisfy { $0.count == 19 })
        XCTAssertEqual(token, TactileToken.empty)
        XCTAssertEqual(token.flattened.count, 608)
        XCTAssertEqual(token.activeRowCount, 0)
    }

    /// Before any env step the reading holds rest rows (stretch 1.0) marked not fresh, and
    /// they give the all-zero token.
    func testReadingBeforeAnyEnvStepIsRestRows() throws {
        let sense = try TactileSense.released(table: lineTable(), tables: modelTables(),
                                              velocity: RigidPointVelocity.atBodyOrigin)
        let reading = try sense.read(TactileFakePhysics(), at: ModelStep.start)
        XCTAssertFalse(reading.isFresh)
        XCTAssertEqual(reading.primary.shape, [1, 10])
        XCTAssertEqual(reading.primary.values, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        XCTAssertEqual(reading.auxiliary?.values, [Float](repeating: 0, count: 10))
        XCTAssertEqual(try PressureSortedPointCloud().build(reading: reading), TactileToken.empty)
    }

    /// A reading of another shape is refused by the token builder.
    func testBuildRefusesAnotherShape() throws {
        let wide = SenseReading(sense: .tactile, primary: ShapedArray(zerosOfShape: [2, 10]),
                                auxiliary: ShapedArray(zerosOfShape: [2, 11]))
        XCTAssertThrowsError(try PressureSortedPointCloud().build(reading: wide))
        let other = SenseReading(sense: .audio, primary: ShapedArray(zerosOfShape: [2, 10]),
                                 auxiliary: ShapedArray(zerosOfShape: [2, 10]))
        XCTAssertThrowsError(try PressureSortedPointCloud().build(reading: other)) { error in
            XCTAssertEqual(error as? TactileReadingError, .wrongSense(.audio))
        }
    }

    /// The worked self-touch of the file header.
    private func selfTouch(distance: Double) -> TactileFakeContactPhysics {
        var p = physics()
        p.contacts = [RawContact(geom1: 3, geom2: 4, position: .zero, frame: zFrame, distance: distance)]
        p.forces = [ContactForce(normal: 1, tangent1: 0, tangent2: 0)]
        p.velocities[5] = Vec3(0, 0, 0.1)
        return p
    }

    /// The three negation rules of ContactReader.swift's header on the worked self-touch.
    func testFrameNegationRules() throws {
        let reader = ContactReader(skinBodyIds: [5, 6], ownRootBodyId: 1)
        let p = selfTouch(distance: -0.002)
        let physical = reader.read(p)
        XCTAssertEqual(physical, reader.read(p, negation: .none))
        XCTAssertEqual(physical.count, 2)
        assertVector(physical[0].normalWorld, 0, 0, -1)
        assertVector(physical[0].positionBody, 0, 0, 0.001)
        XCTAssertEqual(try XCTUnwrap(physical[0].closingSpeed), 0.1, accuracy: 1e-12)
        assertVector(physical[1].normalWorld, 0, 0, 1)
        assertVector(physical[1].positionBody, 0, 0, -0.001)
        XCTAssertEqual(try XCTUnwrap(physical[1].closingSpeed), 0.1, accuracy: 1e-12)

        let second = reader.read(p, negation: .geom2Side)
        XCTAssertEqual(second[0], physical[0])
        assertVector(second[1].normalWorld, 0, 0, -1)
        assertVector(second[1].normalBody, 0, 0, -1)
        assertVector(second[1].positionBody, 0, 0, 0.001)
        XCTAssertEqual(try XCTUnwrap(second[1].closingSpeed), 0.0, accuracy: 0)
        assertVector(second[1].forceBody, 0, 0, 1)

        let first = reader.read(p, negation: .geom1Side)
        XCTAssertEqual(first[1], physical[1])
        assertVector(first[0].normalWorld, 0, 0, 1)
        assertVector(first[0].positionBody, 0, 0, -0.001)
        XCTAssertEqual(try XCTUnwrap(first[0].closingSpeed), 0.0, accuracy: 0)
        assertVector(first[0].forceBody, 0, 0, -1)

        // no penetration: the frame is never negated (touch.py:1409 reads the normal only
        // when dist < 0)
        let touching = selfTouch(distance: 0)
        XCTAssertEqual(reader.read(touching, negation: .geom2Side), reader.read(touching))
        XCTAssertEqual(reader.read(touching, negation: .geom1Side), reader.read(touching))
        // geom 1 on a body without skin (the floor): nothing is negated
        var floor = physics()
        floor.contacts = [RawContact(geom1: 0, geom2: 3, position: .zero, frame: zFrame, distance: -0.002)]
        floor.forces = [ContactForce(normal: 1, tangent1: 0, tangent2: 0)]
        XCTAssertEqual(reader.read(floor, negation: .geom2Side), reader.read(floor))
    }

    /// The reader's cadence applies the rules as the trainer meets them: the pooled records
    /// after a physics-step read negate the geom-2 side; an env step without one negates the
    /// geom-1 side; with the reproduction off both records are physical.
    func testTactileReaderAppliesTheRulesPerCallSequence() throws {
        func onePoint(_ id: Int) throws -> SkinBodyRows {
            let mesh = SkinMeshRows(positions: [Vec3(0, 0, 0)], active: [true], sensorIndex: [0], neighbours: [[]])
            return try XCTUnwrap(SkinBodyRows(bodyId: id, bodyName: "body\(id)", scale: 0.01, submeshes: [mesh]))
        }
        let table = try TactileHandTable(bodies: [onePoint(5), onePoint(6)])
        let p = selfTouch(distance: -0.002)
        let reader = try TactileReader.released(table: table, rootBodyId: 1, ownRootBodyId: 1)
        XCTAssertTrue(reader.reproducesFrameNegation)
        reader.readPhysicsStep(p)
        reader.poolEnvStep(p)
        assertVector(reader.lastContacts[0].normalWorld, 0, 0, -1)
        assertVector(reader.lastContacts[1].normalWorld, 0, 0, -1)
        reader.readEnvStep(p)
        assertVector(reader.lastContacts[0].normalWorld, 0, 0, 1)
        assertVector(reader.lastContacts[1].normalWorld, 0, 0, 1)
        let plain = try TactileReader.released(table: table, rootBodyId: 1, ownRootBodyId: 1,
                                               reproducesFrameNegation: false)
        plain.readEnvStep(p)
        assertVector(plain.lastContacts[0].normalWorld, 0, 0, -1)
        assertVector(plain.lastContacts[1].normalWorld, 0, 0, 1)
        // one point per body: the whole 1 N lands on it whatever the rule
        assertVector(try XCTUnwrap(plain.lastPointForces).forces(ofBody: 6)[0], 0, 0, 1, accuracy: 1e-6)
    }

    /// The worked token of the file header, through the sense's full cadence.
    func testOneContactThroughTheSense() throws {
        let sense = try TactileSense.released(table: lineTable(), tables: modelTables(),
                                              velocity: RigidPointVelocity.atBodyOrigin)
        let fake = TactileFakePhysics()
        fake.contactList = [try Contact(geomIds: [0, 1], distance: 0, position: [0, 0, 0],
                                        frame: [1, 0, 0, 0, 1, 0, 0, 0, 1], force: [10, 0, 0, 0, 0, 0])]
        fake.bodyPositions[3 * 1 + 2] = -0.5   // the hip, the root body, 0.5 m below the palm
        let reading = try runModelStep(sense, fake)
        let token = try PressureSortedPointCloud().build(reading: reading)
        XCTAssertEqual(token.activeRowCount, 1)
        let row = token.rows[0]
        XCTAssertEqual(row[0], 0.0)
        XCTAssertEqual(row[2], 1.0, accuracy: 1e-6)
        XCTAssertEqual(row[3], Float(10.0 / 0.0003 / 100_000.0), accuracy: 1e-5)
        XCTAssertEqual(row[6], 0.015, accuracy: 1e-6)
        // five env steps of 10 N: the history ends in 5 samples of 10 N
        let history = [Double](repeating: 0, count: 95) + [Double](repeating: 10, count: 5)
        let power = Goertzel(frequenciesHz: [5, 10, 15, 20], dt: 0.01, window: 100).power(history)
        XCTAssertEqual(row[7], Float(power[0] / 10.0), accuracy: 1e-6)
        XCTAssertEqual(row[18], 1.0)
        XCTAssertTrue(token.rows[1].allSatisfy { $0 == 0 })
        let forces = try XCTUnwrap(sense.reader.lastPointForces).forces(ofBody: 7)
        // the frame's first row (1, 0, 0) is the normal, so the spread force points along x
        XCTAssertEqual(forces[0].x, 40.0 / 7.0, accuracy: 1e-6)
        sense.reset()
        XCTAssertFalse(try sense.read(fake, at: ModelStep.start).isFresh)
    }

    /// Three active patches sort by normalised pressure, highest first, with the valid flag last;
    /// a zero slot is skipped.
    func testRowsSortByPressure() {
        var slots = [[Float]](repeating: [Float](repeating: 0, count: 10), count: 256)
        slots[0][0] = 0.1
        slots[2][0] = 0.5
        slots[3][0] = 0.3
        let positions = [Vec3(0.5, 0, 0), Vec3(0, 0, 0), Vec3(0, 0.5, 0), Vec3(0, 0, 0.5)]
        let token = PressureSortedPointCloud().build(normalisedSlots: slots, patchPositions: positions,
                                                     selfColumns: nil, contactColumns: nil)
        XCTAssertEqual(token.activeRowCount, 3)
        XCTAssertEqual(token.rows[0][3], 0.5)
        XCTAssertEqual(token.rows[0][1], 1.0)
        XCTAssertEqual(token.rows[1][3], 0.3)
        XCTAssertEqual(token.rows[1][2], 1.0)
        XCTAssertEqual(token.rows[2][3], 0.1)
        XCTAssertEqual(token.rows[2][0], 1.0)
        XCTAssertEqual(token.rows[2][18], 1.0)
        XCTAssertEqual(token.rows[3][18], 0.0)
    }

    /// `_FIELD_NORMS` and the stretch re-zeroing: (15,000, 20,000, 200, 200, 4, 0, 0, 0, 1, 1.2)
    /// gives (0.15, 0.4, 1, 0.01, 0.4, 0, 0, 0, 0, 0.4); padding to 256 slots.
    func testFieldNorms() throws {
        let out = try TactileFieldNorms.normalise([[15_000, 20_000, 200, 200, 4, 0, 0, 0, 1, 1.2]])
        XCTAssertEqual(out.count, 256)
        let expected: [Float] = [0.15, 0.4, 1, 0.01, 0.4, 0, 0, 0, 0, 0.4]
        for c in 0 ..< 10 {
            XCTAssertEqual(out[0][c], expected[c], accuracy: 1e-6)
        }
        XCTAssertEqual(out[1], [Float](repeating: 0, count: 10))
        XCTAssertEqual(TactileFieldNorms.divisors, [100_000, 50_000, 200, 20_000, 10, 10, 10, 10, 0.5, 0.5])
        XCTAssertThrowsError(try TactileFieldNorms.normalise([[1, 2, 3]]))
    }
}
