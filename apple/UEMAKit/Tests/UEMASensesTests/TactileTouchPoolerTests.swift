// TactileTouchPoolerTests.swift: hand-computed checks of the patch fields (work item WI-28;
// GoertzelTouchPooler and Goertzel of TouchPooler.swift), each the Swift form of a test in
// Source/Python/mimo_bridge/tests/test_touch_adapter.py.
//
// The worked example (test_touch_adapter.py:4-18), two skin bodies, dt 0.01 s:
//   body 7 "palm": 4 points, spacing 0.01 m  -> one point's area 100 mm2
//   body 9 "tip":  3 points, spacing 0.002 m -> one point's area 4 mm2
//   body 3 "eye":  1 point, spacing 1.0 m    -> not skin (spacing at or above 0.5 m), dropped
// Update 1 on the palm: one contact, normal force 3 N, tangential (4, 0, 0) N, two of the
// four points carry force:
//   area 2 x 100 mm2 = 200 mm2 = 0.0002 m2; pressure 3 / 0.0002 = 15,000 Pa;
//   shear 4 / 0.0002 = 20,000 Pa; slip 0 mm/s (first sight).
// Update 2: the same contact 2 mm farther along the skin: slip 0.002 m / 0.01 s = 200 mm/s.
// The fields are float32 (touch_adapter.py:345 `fields.astype(np.float32)`).
// Nothing in this file was compiled on the machine that wrote it (no Swift toolchain).

import Foundation
import XCTest
import UEMASenses

final class TactileTouchPoolerTests: XCTestCase {
    private let dt = 0.01

    private func specs() -> [PatchSpec] {
        [
            PatchSpec(bodyId: 9, bodyName: "tip", pointCount: 3, scale: 0.002),
            PatchSpec(bodyId: 7, bodyName: "palm", pointCount: 4, scale: 0.01),
            PatchSpec(bodyId: 3, bodyName: "eye", pointCount: 1, scale: 1.0),
        ]
    }

    /// Point forces with the given per-body rows (N, body frame).
    private func forces(_ rows: [Int: [Vec3]]) -> SkinPointForces {
        var counts: [Int: Int] = [:]
        for (body, list) in rows { counts[body] = list.count }
        var out = SkinPointForces(pointCounts: counts)
        for (body, list) in rows {
            for (i, f) in list.enumerated() { out.set(f, body: body, point: i) }
        }
        return out
    }

    private func palmForces(_ zs: [Double]) -> SkinPointForces {
        forces([7: zs.map { Vec3(0, 0, $0) }])
    }

    /// One contact record (test_touch_adapter.py:42-47 `contact`).
    private func contact(body: Int, normal: Double, tangential: Vec3 = .zero, positionBody: Vec3 = .zero,
                         pair: (Int, Int) = (1, 2), penetration: Double = 0, closing: Double = 0,
                         normalBody: Vec3 = Vec3(0, 0, 1), otherName: String = "",
                         otherIsSelf: Bool = false) -> SkinContact {
        SkinContact(bodyId: body, geom1: pair.0, geom2: pair.1, normalForce: normal,
                    tangentialBody: tangential, positionBody: positionBody, positionWorld: .zero,
                    normalWorld: Vec3(0, 0, 1), normalBody: normalBody,
                    forceBody: normalBody * normal + tangential, penetration: penetration,
                    closingSpeed: closing, tangentSpeed: nil, frictionPower: nil,
                    otherBodyName: otherName, otherIsSelf: otherIsSelf)
    }

    /// test_patch_order_and_eye_dropped: body id order, the eye dropped, point areas 100 and 4 mm2.
    func testPatchOrderAndEyeDropped() throws {
        let pooler = try GoertzelTouchPooler(specs: specs(), dt: dt)
        XCTAssertEqual(pooler.patchCount, 2)
        XCTAssertEqual(pooler.specs.map { $0.bodyId }, [7, 9])
        XCTAssertEqual(pooler.pointAreaMm2[0], 100.0, accuracy: 1e-9)
        XCTAssertEqual(pooler.pointAreaMm2[1], 4.0, accuracy: 1e-9)
        XCTAssertNil(pooler.patchRow(ofBody: 3))
    }

    /// test_pressure_shear_area_by_hand: 15,000 Pa, 20,000 Pa, 200 mm2, slip 0, stretch 1.0.
    func testPressureShearAreaByHand() throws {
        let pooler = try GoertzelTouchPooler(specs: specs(), dt: dt)
        var pf = palmForces([2, 1, 0, 0])
        pf = forces([7: pf.forces(ofBody: 7), 9: [Vec3](repeating: .zero, count: 3)])
        let out = pooler.update(pointForces: pf, contacts: [contact(body: 7, normal: 3, tangential: Vec3(4, 0, 0))],
                                includeVibration: true)
        XCTAssertEqual(out.patchCount, 2)
        let palm = out.fields[0]
        XCTAssertEqual(palm.count, 10)
        XCTAssertEqual(palm[3], 200.0, accuracy: 1e-3)
        XCTAssertEqual(palm[0], 15_000.0, accuracy: 1e-2)
        XCTAssertEqual(palm[1], 20_000.0, accuracy: 1e-2)
        XCTAssertEqual(palm[2], 0.0)
        XCTAssertEqual(palm[8], 1.0)
        XCTAssertEqual(palm[9], 1.0)
        XCTAssertEqual(out.fields[1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    }

    /// test_slip_by_hand: 2 mm in 0.01 s is 200 mm/s; a different pair inherits no slip.
    func testSlipByHand() throws {
        let pooler = try GoertzelTouchPooler(specs: specs(), dt: dt)
        let pf = palmForces([1, 0, 0, 0])
        _ = pooler.update(pointForces: pf, contacts: [contact(body: 7, normal: 1)], includeVibration: true)
        let second = pooler.update(pointForces: pf, contacts: [contact(body: 7, normal: 1, positionBody: Vec3(0.002, 0, 0))],
                                   includeVibration: true)
        XCTAssertEqual(second.fields[0][2], 200.0, accuracy: 1e-3)
        let third = pooler.update(pointForces: pf, contacts: [contact(body: 7, normal: 1, positionBody: Vec3(0.1, 0, 0), pair: (5, 6))],
                                  includeVibration: true)
        XCTAssertEqual(third.fields[0][2], 0.0)
    }

    /// test_no_area_means_no_pressure: no point carries force, so area 0, pressure 0, no NaN.
    func testNoAreaMeansNoPressure() throws {
        let pooler = try GoertzelTouchPooler(specs: specs(), dt: dt)
        let out = pooler.update(pointForces: palmForces([0, 0, 0, 0]),
                                contacts: [contact(body: 7, normal: 3, tangential: Vec3(4, 0, 0))],
                                includeVibration: true)
        XCTAssertEqual(out.fields[0][0], 0.0)
        XCTAssertEqual(out.fields[0][1], 0.0)
        XCTAssertEqual(out.fields[0][3], 0.0)
        XCTAssertFalse(out.fields.flatMap { $0 }.contains { $0.isNaN })
    }

    /// test_goertzel_by_hand: a 10 Hz sine of amplitude 2 N over 100 samples at 0.01 s gives
    /// power 4.0 N^2 in the 10 Hz band and under 1e-9 N^2 in the others; a constant 5 N gives
    /// under 1e-9 N^2 in every band.
    func testGoertzelByHand() {
        let g = Goertzel(frequenciesHz: [5, 10, 15, 20], dt: 0.01, window: 100)
        let sine = (0 ..< 100).map { 2.0 * sin(2.0 * Double.pi * 10.0 * Double($0) * 0.01) }
        let p = g.power(sine)
        XCTAssertEqual(p[1], 4.0, accuracy: 1e-9)
        XCTAssertLessThan(abs(p[0]), 1e-9)
        XCTAssertLessThan(abs(p[2]), 1e-9)
        XCTAssertLessThan(abs(p[3]), 1e-9)
        let flat = g.power([Double](repeating: 5.0, count: 100))
        XCTAssertTrue(flat.allSatisfy { abs($0) < 1e-9 })
        XCTAssertEqual(g.power([]), [0, 0, 0, 0])
    }

    /// test_vibration_bands_through_pooler: 3 N + 1 N x sin(2 pi 20 Hz t) over one full
    /// window gives 1.0 N^2 in the 20 Hz band and under 1e-6 N^2 in the other three.
    func testVibrationBandsThroughPooler() throws {
        let pooler = try GoertzelTouchPooler(specs: specs(), dt: dt, vibrationWindow: 100)
        let pf = palmForces([1, 0, 0, 0])
        var last: TouchFields?
        for i in 0 ..< 100 {
            let n = 3.0 + sin(2.0 * Double.pi * 20.0 * Double(i) * dt)
            last = pooler.update(pointForces: pf, contacts: [contact(body: 7, normal: n)], includeVibration: true)
        }
        let palm = try XCTUnwrap(last).fields[0]
        XCTAssertEqual(palm[7], 1.0, accuracy: 1e-6)
        XCTAssertLessThan(abs(palm[4]), 1e-6)
        XCTAssertLessThan(abs(palm[5]), 1e-6)
        XCTAssertLessThan(abs(palm[6]), 1e-6)
    }

    /// One sample of 3 N in a 100-sample window: power 4 x 3^2 / 100^2 = 0.0036 N^2 in every
    /// band (the Goertzel recurrence of one final sample v gives real v, imag 0).
    func testOneSampleHistoryGivesTheSamePowerInEveryBand() throws {
        let pooler = try GoertzelTouchPooler(specs: specs(), dt: dt)
        let out = pooler.update(pointForces: palmForces([1, 0, 0, 0]), contacts: [contact(body: 7, normal: 3)],
                                includeVibration: true)
        for b in 4 ..< 8 {
            XCTAssertEqual(out.fields[0][b], 0.0036, accuracy: 1e-8)
        }
        let skipped = try GoertzelTouchPooler(specs: specs(), dt: dt)
            .update(pointForces: palmForces([1, 0, 0, 0]), contacts: [contact(body: 7, normal: 3)], includeVibration: false)
        XCTAssertEqual(Array(skipped.fields[0][4 ..< 8]), [0, 0, 0, 0])
    }

    /// test_too_many_skin_bodies_rejected: 257 bodies exceed the 256 patch slots.
    func testTooManySkinBodiesRejected() {
        let many = (0 ..< 257).map { PatchSpec(bodyId: $0, bodyName: "b\($0)", pointCount: 1, scale: 0.01) }
        XCTAssertThrowsError(try GoertzelTouchPooler(specs: many, dt: dt)) { error in
            XCTAssertEqual(error as? TouchPoolerError, .tooManyPatches(count: 257, maximum: 256))
        }
    }

    /// test_own_part_force_and_own_part_count_are_accumulated_per_patch: 3 N from an own part
    /// and 1 N from the floor give share 0.75 and parts 1 / 4 = 0.25; a second own part gives
    /// 5 / 6 and 0.5; six own parts give 1.0 and 1.0 (capped at 4 parts).
    func testOwnPartForceAndCount() throws {
        let pooler = try GoertzelTouchPooler(specs: specs(), dt: dt)
        let pf = palmForces([2, 1, 0, 0])
        var out = pooler.update(pointForces: pf, contacts: [
            contact(body: 7, normal: 3, pair: (1, 2), otherName: "left_th", otherIsSelf: true),
            contact(body: 7, normal: 1, pair: (3, 4), otherName: "floor"),
        ], includeVibration: true)
        XCTAssertEqual(out.selfColumns[0][0], 0.75, accuracy: 1e-6)
        XCTAssertEqual(out.selfColumns[0][1], 0.25, accuracy: 1e-6)
        XCTAssertEqual(out.selfColumns[1], [0, 0])
        XCTAssertEqual(out.selfForceTotal, 3.0, accuracy: 1e-12)
        XCTAssertEqual(out.totalForce, 4.0, accuracy: 1e-12)
        XCTAssertEqual(out.selfPairCount, 1)
        out = pooler.update(pointForces: pf, contacts: [
            contact(body: 7, normal: 3, pair: (1, 2), otherName: "left_th", otherIsSelf: true),
            contact(body: 7, normal: 2, pair: (5, 6), otherName: "right_th", otherIsSelf: true),
            contact(body: 7, normal: 1, pair: (3, 4), otherName: "floor"),
        ], includeVibration: true)
        XCTAssertEqual(out.selfColumns[0][0], Float(5.0 / 6.0), accuracy: 1e-6)
        XCTAssertEqual(out.selfColumns[0][1], 0.5, accuracy: 1e-6)
        XCTAssertEqual(out.selfPairCount, 2)
        out = pooler.update(pointForces: pf, contacts: (0 ..< 6).map {
            contact(body: 7, normal: 1, pair: (10 + $0, 20 + $0), otherName: "own_\($0)", otherIsSelf: true)
        }, includeVibration: true)
        XCTAssertEqual(out.selfColumns[0][0], 1.0, accuracy: 1e-6)
        XCTAssertEqual(out.selfColumns[0][1], 1.0, accuracy: 1e-6)
    }

    /// test_one_self_touch_is_one_pair_even_though_both_sides_feel_it.
    func testOneSelfTouchIsOnePair() throws {
        let pooler = try GoertzelTouchPooler(specs: specs(), dt: dt)
        let pf = forces([7: [Vec3(0, 0, 1), .zero, .zero, .zero], 9: [Vec3(0, 0, 1), .zero, .zero]])
        let out = pooler.update(pointForces: pf, contacts: [
            contact(body: 7, normal: 2, pair: (4, 8), otherName: "tip", otherIsSelf: true),
            contact(body: 9, normal: 2, pair: (4, 8), otherName: "palm", otherIsSelf: true),
        ], includeVibration: true)
        XCTAssertEqual(out.selfPairCount, 1)
        XCTAssertEqual(out.selfColumns[0][1], 0.25, accuracy: 1e-6)
        XCTAssertEqual(out.selfColumns[1][1], 0.25, accuracy: 1e-6)
    }

    /// test_a_patch_with_no_force_has_no_self_share_and_raises_nothing: forces 0 N and -5 N.
    func testZeroAndNegativeForceGiveNoNaN() throws {
        let pooler = try GoertzelTouchPooler(specs: specs(), dt: dt)
        let out = pooler.update(pointForces: palmForces([0, 0, 0, 0]), contacts: [
            contact(body: 7, normal: 0, otherName: "left_th", otherIsSelf: true),
            contact(body: 7, normal: -5, pair: (7, 8), otherName: "right_th", otherIsSelf: true),
        ], includeVibration: true)
        XCTAssertFalse(out.selfColumns.flatMap { $0 }.contains { $0.isNaN })
        XCTAssertFalse(out.fields.flatMap { $0 }.contains { $0.isNaN })
    }

    /// The push columns of the strongest contact: depth 0.001 m = 0.1 cm reads 1.0 (scale
    /// 0.1 cm); closing speed 0.04 m/s = 4 cm/s reads 2.0 (scale 2 cm/s); direction in the
    /// body frame (touch_adapter.py:279-288).
    func testPushColumnsFollowTheStrongestContact() throws {
        let pooler = try GoertzelTouchPooler(specs: specs(), dt: dt)
        let out = pooler.update(pointForces: palmForces([1, 0, 0, 0]), contacts: [
            contact(body: 7, normal: 1, pair: (1, 2), penetration: 0.005, closing: 0.5, normalBody: Vec3(1, 0, 0)),
            contact(body: 7, normal: 4, pair: (3, 4), penetration: 0.001, closing: 0.04, normalBody: Vec3(0, 1, 0)),
        ], includeVibration: true)
        let c = out.contactColumns[0]
        XCTAssertEqual(c[0], 0.0)
        XCTAssertEqual(c[1], 1.0)
        XCTAssertEqual(c[3], 1.0, accuracy: 1e-6)
        XCTAssertEqual(c[4], 2.0, accuracy: 1e-6)
        XCTAssertEqual(out.contactColumns[1], [0, 0, 0, 0, 0])
    }

    /// reset() returns the pooler to a fresh one: no slip from a position stored before it.
    func testResetForgetsStoredPositions() throws {
        let pooler = try GoertzelTouchPooler(specs: specs(), dt: dt)
        let pf = palmForces([1, 0, 0, 0])
        _ = pooler.update(pointForces: pf, contacts: [contact(body: 7, normal: 1)], includeVibration: true)
        pooler.reset()
        XCTAssertNil(pooler.lastFields)
        let out = pooler.update(pointForces: pf, contacts: [contact(body: 7, normal: 1, positionBody: Vec3(0.002, 0, 0))],
                                includeVibration: true)
        XCTAssertEqual(out.fields[0][2], 0.0)
        XCTAssertEqual(out.fields[0][4], 4.0 * 1.0 / 10_000.0, accuracy: 1e-9)
    }
}
