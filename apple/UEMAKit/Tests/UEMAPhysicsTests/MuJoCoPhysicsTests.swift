// MuJoCoPhysicsTests.swift: the tests of MuJoCoPhysics on the released crib scene (work item
// WI-25 of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html). The acceptance line of the
// item is testTwentyModelStepsAdvanceOneSecondWithFiniteQpos: "loads the WI-15 bundle, steps 20
// model steps (1.0 s); nq equals 146; simulated time advanced 1.0 s; no NaN in qpos".
//
// INPUT. The scene.xml of the shipped bundle folder, found from this file's own path:
// #filePath is Tests/UEMAPhysicsTests/<this file>, so the package root apple/UEMAKit is three
// folders up, and the scene is Sources/UEMAScene/Resources/crib_age24_trunk_v1/scene.xml there.
// This test target depends on UEMAPhysics, not UEMAScene (Package.swift), so it reads the file by
// path instead of through ResourceSceneBundle. Runs on the macOS destination, where the source
// tree is on disk.
//
// THE NUMBERS COMPARED AGAINST were measured with Python mujoco 3.3.0 on the same scene.xml on
// November 4, 2026 (the MEASURED ON THE RELEASED SCENE block of MuJoCoPhysics.swift's header):
// nq 146, nv 136, nu 90, nbody 76, njnt 104, nsensor 56, nsensordata 168, ngeom 188, ncam 6,
// timestep 0.005 s; 200 physics steps of zero control end at time 1.0000000000000007 s with every
// qpos finite; 182 geoms from mjv_updateScene at the initial pose; camera ids eye_left 1,
// fovea_left 2, ring_left 3, eye_right 4, fovea_right 5; body left_eye 45; force-range low end
// above -1 on 42 motors and high end below 1 on 14; geom_bodyid 188 entries, body_rootid 76
// entries with body_rootid[45] = 1. Two small scenes written by the tests themselves (a malformed
// file and a one-body scene at timestep 0.002 s) were loaded with the same Python mujoco 3.3.0:
// the first raises "XML parse error 14", the second compiles with opt.timestep 0.002, nq 7, nu 0;
// the same body at timestep 0.005 s compiles with nq 7, nu 0, nbody 2.
//
// 18 tests. Nothing here was compiled on the machine that wrote it (no Swift toolchain).

import Foundation
import UEMACore
import UEMAPhysics
import XCTest

final class MuJoCoPhysicsTests: XCTestCase {
    /// apple/UEMAKit/Sources/UEMAScene/Resources/crib_age24_trunk_v1/scene.xml.
    static var sceneURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // UEMAPhysicsTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // UEMAKit
            .appendingPathComponent("Sources/UEMAScene/Resources/crib_age24_trunk_v1/scene.xml")
    }

    /// A physics with the released scene loaded.
    func loadedPhysics() throws -> MuJoCoPhysics {
        let physics = MuJoCoPhysics()
        try physics.load(sceneXML: Self.sceneURL)
        return physics
    }

    /// Writes `xml` to a new file in a new temporary folder, removed after the test.
    func temporaryScene(_ xml: String, named name: String = "scene.xml") throws -> URL {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("UEMAPhysicsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        addTeardownBlock {
            _ = try? FileManager.default.removeItem(at: folder)
        }
        let url = folder.appendingPathComponent(name)
        try xml.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    /// One free body with a sphere 1 m above the floor, physics timestep `timestep` seconds.
    static func oneBodyXML(timestep: String) -> String {
        "<mujoco><option timestep=\"\(timestep)\"/><worldbody><body pos=\"0 0 1\"><freejoint/>"
            + "<geom size=\"0.1\"/></body></worldbody></mujoco>"
    }

    // MARK: Loading

    /// The model sizes equal the Python measurement of the same file.
    func testLoadsReleasedSceneWithMeasuredSizes() throws {
        let physics = try loadedPhysics()
        XCTAssertTrue(physics.isLoaded)
        XCTAssertEqual(physics.sizes, PhysicsSizes(nq: 146, nv: 136, nu: 90, nbody: 76, njnt: 104,
                                                   nsensordata: 168, ngeom: 188, ncam: 6))
        XCTAssertEqual(physics.timestep, 0.005, accuracy: 1e-15)
        XCTAssertEqual(physics.time, 0)
        XCTAssertEqual(physics.qpos.count, 146)
        XCTAssertEqual(physics.qvel.count, 136)
        XCTAssertEqual(physics.xpos.count, 76 * 3)
        XCTAssertEqual(physics.xquat.count, 76 * 4)
        XCTAssertEqual(physics.xmat.count, 76 * 9)
        XCTAssertEqual(physics.xaxis.count, 104 * 3)
        XCTAssertEqual(physics.xanchor.count, 104 * 3)
        XCTAssertEqual(physics.sensordata.count, 168)
        XCTAssertEqual(physics.actuatorForce.count, 90)
        XCTAssertEqual(physics.control.count, 90)
        XCTAssertEqual(physics.sceneURL, Self.sceneURL)
    }

    /// The acceptance line: 20 model steps of zero control advance simulated time by 1.0 s,
    /// call the per-physics-step observer 200 times (10 per model step), and leave every qpos finite.
    func testTwentyModelStepsAdvanceOneSecondWithFiniteQpos() throws {
        let physics = try loadedPhysics()
        XCTAssertEqual(physics.sizes.nq, 146)
        let startTime = physics.time
        let zero = [Double](repeating: 0, count: physics.sizes.nu)
        var physicsStepCallbacks = 0
        for _ in 0 ..< 20 {
            try physics.stepModelStep(controlValues: zero) { reading in
                XCTAssertTrue(reading === physics)
                physicsStepCallbacks += 1
            }
        }
        XCTAssertEqual(physics.time - startTime, 20 * ModelStep.seconds, accuracy: 1e-9)
        XCTAssertEqual(physics.time - startTime, 1.0, accuracy: 1e-9)
        XCTAssertEqual(physicsStepCallbacks, 200)
        XCTAssertEqual(physics.physicsStepCount, 200)
        let qpos = physics.qpos
        XCTAssertEqual(qpos.count, 146)
        XCTAssertTrue(qpos.allSatisfy { $0.isFinite }, "non-finite qpos at \(qpos.indices.filter { !qpos[$0].isFinite })")
        XCTAssertEqual(physics.state.nonFiniteFields, [])
    }

    /// A missing scene file throws sceneFileMissing with its path, and a step before any load
    /// throws notLoaded.
    func testMissingFileAndUnloadedStepThrow() {
        let physics = MuJoCoPhysics()
        XCTAssertFalse(physics.isLoaded)
        XCTAssertEqual(physics.qpos, [])
        XCTAssertEqual(physics.sizes.nq, 0)
        let missing = URL(fileURLWithPath: "/nonexistent/folder/scene.xml")
        XCTAssertThrowsError(try physics.load(sceneXML: missing)) { error in
            XCTAssertEqual(error as? PhysicsError, .sceneFileMissing(path: missing.path))
        }
        XCTAssertThrowsError(try physics.stepPhysicsStep()) { error in
            XCTAssertEqual(error as? PhysicsError, .notLoaded)
        }
        XCTAssertThrowsError(try physics.modelTables()) { error in
            XCTAssertEqual(error as? PhysicsError, .notLoaded)
        }
    }

    /// A malformed scene file throws loadFailed carrying MuJoCo's parse message, and leaves no
    /// scene loaded.
    func testMalformedSceneThrowsLoadFailedWithMessage() throws {
        let url = try temporaryScene("<mujoco><worldbody><body></mujoco>")
        let physics = MuJoCoPhysics()
        XCTAssertThrowsError(try physics.load(sceneXML: url)) { error in
            guard case let .loadFailed(path, message)? = error as? PhysicsError else {
                return XCTFail("expected loadFailed, got \(error)")
            }
            XCTAssertEqual(path, url.path)
            XCTAssertTrue(message.contains("XML parse error"), message)
        }
        XCTAssertFalse(physics.isLoaded)
    }

    /// A scene whose timestep is not 0.005 s throws timestepMismatch with both values: a model
    /// step of 10 physics steps would then not span 0.050 s.
    func testOtherTimestepThrowsTimestepMismatch() throws {
        let url = try temporaryScene(Self.oneBodyXML(timestep: "0.002"))
        let physics = MuJoCoPhysics()
        XCTAssertThrowsError(try physics.load(sceneXML: url)) { error in
            XCTAssertEqual(error as? PhysicsError, .timestepMismatch(expectedSeconds: 0.005, foundSeconds: 0.002))
        }
        XCTAssertFalse(physics.isLoaded)
    }

    /// A second load replaces the first: sizes, clip bounds, step count and scene URL are those of
    /// the new scene; one env step there runs 2 physics steps and advances 0.010 s.
    func testReloadReplacesSceneAndEnvStepRunsTwoPhysicsSteps() throws {
        let physics = try loadedPhysics()
        try physics.stepModelStep(controlValues: [Double](repeating: 0, count: physics.sizes.nu))
        let url = try temporaryScene(Self.oneBodyXML(timestep: "0.005"))
        try physics.load(sceneXML: url)
        XCTAssertEqual(physics.sceneURL, url)
        XCTAssertEqual(physics.sizes.nq, 7)
        XCTAssertEqual(physics.sizes.nu, 0)
        XCTAssertEqual(physics.sizes.nbody, 2)
        XCTAssertEqual(physics.controlLowerBounds, [])
        XCTAssertEqual(physics.physicsStepCount, 0)
        XCTAssertEqual(physics.time, 0)
        var callbacks = 0
        try physics.stepEnvStep(controlValues: []) { _ in callbacks += 1 }
        XCTAssertEqual(callbacks, ModelStep.physicsStepsPerEnvStep)
        XCTAssertEqual(physics.physicsStepCount, 2)
        XCTAssertEqual(physics.time, ModelStep.envStepSeconds, accuracy: 1e-12)
    }

    // MARK: Controls

    /// Controls far outside the range land on the per-motor clip bounds, which lie within -1..1
    /// and equal the compiled actuator_forcerange, rounded through Float, clipped to -1..1.
    func testWriteClipsToMotorBounds() throws {
        let physics = try loadedPhysics()
        let nu = physics.sizes.nu
        let tables = try physics.modelTables()
        try physics.write(controls: [Double](repeating: 5, count: nu))
        XCTAssertEqual(physics.control, physics.controlUpperBounds)
        try physics.write(controls: [Double](repeating: -5, count: nu))
        XCTAssertEqual(physics.control, physics.controlLowerBounds)
        for motor in 0 ..< nu {
            // float32 rounding of the bounds as MIMo's action space stores them (actuation.py:136)
            XCTAssertEqual(physics.controlUpperBounds[motor], min(1, Double(Float(tables.actuatorForceRangeHigh[motor]))))
            XCTAssertEqual(physics.controlLowerBounds[motor], max(-1, Double(Float(tables.actuatorForceRangeLow[motor]))))
            XCTAssertTrue(MotorControls.range.contains(physics.controlLowerBounds[motor]))
            XCTAssertTrue(MotorControls.range.contains(physics.controlUpperBounds[motor]))
        }
        XCTAssertEqual(physics.controlUpperBounds.filter { $0 < 1 }.count, 14)
        XCTAssertEqual(physics.controlLowerBounds.filter { $0 > -1 }.count, 42)
        let inside = (0 ..< nu).map { physics.controlLowerBounds[$0] * 0.5 + physics.controlUpperBounds[$0] * 0.5 }
        try physics.write(controls: inside)
        XCTAssertEqual(physics.control, inside)
    }

    /// A control array of the wrong count or holding a NaN throws and leaves ctrl unchanged.
    func testWriteRejectsWrongCountAndNonFinite() throws {
        let physics = try loadedPhysics()
        let nu = physics.sizes.nu
        XCTAssertThrowsError(try physics.write(controls: [Double](repeating: 0, count: nu + 1))) { error in
            XCTAssertEqual(error as? PhysicsError, .wrongCount(name: "controls", expected: nu, found: nu + 1))
        }
        var withNaN = [Double](repeating: 0.1, count: nu)
        withNaN[7] = .nan
        XCTAssertThrowsError(try physics.write(controls: withNaN)) { error in
            XCTAssertEqual(error as? PhysicsError, .nonFinite(name: "controls", index: 7))
        }
        XCTAssertEqual(physics.control, [Double](repeating: 0, count: nu))
    }

    /// UEMACore's MotorControls (92 numbers) is written when the model has 92 motors and throws
    /// wrongCount otherwise; the released scene has 90 (MuJoCoPhysics.swift, THE MOTOR COUNT).
    func testMotorControlsWriteMatchesTheModelCount() throws {
        let physics = try loadedPhysics()
        let nu = physics.sizes.nu
        if nu == MotorControls.count {
            XCTAssertNoThrow(try physics.write(controls: MotorControls.zero))
        } else {
            XCTAssertThrowsError(try physics.write(controls: MotorControls.zero)) { error in
                XCTAssertEqual(error as? PhysicsError,
                               .wrongCount(name: "controls", expected: nu, found: MotorControls.count))
            }
            // The Physics protocol's default model step writes first, so it throws before any
            // physics step runs.
            XCTAssertThrowsError(try physics.stepModelStep(controls: MotorControls.zero, afterEachPhysicsStep: { _ in }))
            XCTAssertEqual(physics.physicsStepCount, 0)
            XCTAssertEqual(physics.time, 0)
        }
    }

    // MARK: State

    /// reset() returns time to 0 and qpos to qpos0 after steps.
    func testResetReturnsTimeToZeroAndQposToQpos0() throws {
        let physics = try loadedPhysics()
        let zero = [Double](repeating: 0, count: physics.sizes.nu)
        try physics.stepModelStep(controlValues: zero)
        XCTAssertEqual(physics.time, ModelStep.seconds, accuracy: 1e-12)
        try physics.reset()
        XCTAssertEqual(physics.time, 0)
        XCTAssertEqual(physics.physicsStepCount, 0)
        XCTAssertEqual(physics.qpos, try physics.modelTables().qpos0)
        XCTAssertEqual(physics.qvel, [Double](repeating: 0, count: physics.sizes.nv))
    }

    /// setState writes qpos and qvel; wrong counts throw.
    func testSetStateRoundTripAndCountChecks() throws {
        let physics = try loadedPhysics()
        var qpos = physics.qpos
        qpos[2] += 0.01
        let qvel = (0 ..< physics.sizes.nv).map { Double($0) * 1e-4 }
        try physics.setState(qpos: qpos, qvel: qvel)
        XCTAssertEqual(physics.qpos, qpos)
        XCTAssertEqual(physics.qvel, qvel)
        XCTAssertThrowsError(try physics.setState(qpos: [0], qvel: qvel)) { error in
            XCTAssertEqual(error as? PhysicsError, .wrongCount(name: "qpos", expected: 146, found: 1))
        }
    }

    // MARK: Names, cameras, geoms

    /// Name lookups give the measured ids; an absent name gives nil.
    func testNameLookupsGiveMeasuredIds() throws {
        let physics = try loadedPhysics()
        XCTAssertEqual(physics.id(of: .camera, named: "eye_left"), 1)
        XCTAssertEqual(physics.id(of: .camera, named: "fovea_left"), 2)
        XCTAssertEqual(physics.id(of: .camera, named: "ring_left"), 3)
        XCTAssertEqual(physics.id(of: .camera, named: "eye_right"), 4)
        XCTAssertEqual(physics.id(of: .camera, named: "fovea_right"), 5)
        XCTAssertEqual(physics.id(of: .body, named: "left_eye"), 45)
        XCTAssertEqual(physics.id(of: .actuator, named: "act:hip_bend"), 0)
        XCTAssertEqual(physics.id(of: .sensor, named: "vestibular_acc"), 0)
        XCTAssertNil(physics.id(of: .body, named: "no_such_body"))
    }

    /// Every eye camera has a pose with an orthonormal rotation, and the scene gives the measured
    /// 182 geoms at the initial pose, each with 3 size, 3 position, 9 rotation and 4 colour numbers.
    func testCameraPosesAndSceneGeoms() throws {
        let physics = try loadedPhysics()
        for camera in EyeCamera.allCases {
            let pose = try physics.cameraPose(camera)
            for row in 0 ..< 3 {
                let r = pose.rotation[(row * 3) ..< (row * 3 + 3)]
                XCTAssertEqual(r.reduce(0) { $0 + $1 * $1 }, 1, accuracy: 1e-9, "\(camera) row \(row)")
            }
            XCTAssertTrue(pose.position.allSatisfy { $0.isFinite })
        }
        let geoms = try physics.sceneGeoms()
        XCTAssertEqual(geoms.count, 182)
        XCTAssertTrue(geoms.allSatisfy { $0.rgba.count == 4 && $0.rotation.count == 9 })
        XCTAssertEqual(try physics.sceneGeoms().count, 182, "a second call reuses the buffer")
    }

    // MARK: Contacts and velocities

    /// After 1.0 s every contact has finite numbers, a unit normal and the force of
    /// contactForce(_:) for its id.
    func testContactsAfterOneSecondAreWellFormed() throws {
        let physics = try loadedPhysics()
        let zero = [Double](repeating: 0, count: physics.sizes.nu)
        for _ in 0 ..< 20 {
            try physics.stepModelStep(controlValues: zero)
        }
        let contacts = physics.contacts
        XCTAssertEqual(contacts.count, physics.contactCount)
        XCTAssertGreaterThan(contacts.count, 0, "Python measured 28 contacts at this instant")
        for (id, contact) in contacts.enumerated() {
            let normal = contact.frame[0 ..< 3]
            XCTAssertEqual(normal.reduce(0) { $0 + $1 * $1 }, 1, accuracy: 1e-9)
            XCTAssertTrue(contact.force.allSatisfy { $0.isFinite })
            XCTAssertEqual(contact.force, try physics.contactForce(id))
        }
        XCTAssertThrowsError(try physics.contactForce(contacts.count)) { error in
            XCTAssertEqual(error as? PhysicsError, .contactOutOfRange(id: contacts.count, count: contacts.count))
        }
    }

    /// bodyVelocity gives 6 finite numbers, all zero at rest after load.
    func testBodyVelocityIsSixNumbersAndZeroAtRest() throws {
        let physics = try loadedPhysics()
        let velocity = physics.bodyVelocity(bodyId: 45)
        XCTAssertEqual(velocity.count, 6)
        XCTAssertEqual(velocity, [Double](repeating: 0, count: 6))
    }

    // MARK: Model description, modulation, scratch kinematics

    /// The tables have the model's counts; every control range is -1..1; every gain is fixed and
    /// bias-free (the condition env.py:2835-2836 checks before modulating).
    func testModelTablesCounts() throws {
        let tables = try loadedPhysics().modelTables()
        XCTAssertEqual(tables.bodyCount, 76)
        XCTAssertEqual(tables.jointCount, 104)
        XCTAssertEqual(tables.actuatorCount, 90)
        XCTAssertEqual(tables.sensorCount, 56)
        XCTAssertEqual(tables.sensorDimension.reduce(0, +), 168)
        XCTAssertEqual(tables.qpos0.count, 146)
        XCTAssertEqual(tables.cameraNames, ["watch_cam", "eye_left", "fovea_left", "ring_left", "eye_right", "fovea_right"])
        XCTAssertEqual(tables.actuatorNames.first, "act:hip_bend")
        XCTAssertTrue(tables.actuatorControlRangeLow.allSatisfy { $0 == -1 })
        XCTAssertTrue(tables.actuatorControlRangeHigh.allSatisfy { $0 == 1 })
        XCTAssertTrue(tables.actuatorGainType.allSatisfy { $0 == 0 })
        XCTAssertTrue(tables.actuatorBiasType.allSatisfy { $0 == 0 })
        XCTAssertEqual(tables.timestep, 0.005, accuracy: 1e-15)
        XCTAssertEqual(tables.geomCount, 188)
        XCTAssertTrue(tables.geomBodyID.allSatisfy { (0 ..< 76).contains($0) })
        XCTAssertEqual(tables.bodyRootID.count, 76)
        XCTAssertEqual(tables.bodyRootID[0], 0)
        XCTAssertEqual(tables.bodyRootID[45], 1)
    }

    /// The modulation writes reach the model and leave the control clip bounds as loaded.
    func testActuatorParameterWritesRoundTrip() throws {
        let physics = try loadedPhysics()
        let before = try physics.modelTables()
        let upperBefore = physics.controlUpperBounds
        let halfGain = before.actuatorGainParameter0.map { $0 * 0.5 }
        try physics.setActuatorGainParameter0(halfGain)
        let halfLow = before.actuatorForceRangeLow.map { $0 * 0.5 }
        let halfHigh = before.actuatorForceRangeHigh.map { $0 * 0.5 }
        try physics.setActuatorForceRange(low: halfLow, high: halfHigh)
        let after = try physics.modelTables()
        XCTAssertEqual(after.actuatorGainParameter0, halfGain)
        XCTAssertEqual(after.actuatorForceRangeLow, halfLow)
        XCTAssertEqual(after.actuatorForceRangeHigh, halfHigh)
        XCTAssertEqual(physics.controlUpperBounds, upperBefore)
        XCTAssertThrowsError(try physics.setActuatorGainParameter0([1])) { error in
            XCTAssertEqual(error as? PhysicsError, .wrongCount(name: "actuatorGainParameter0", expected: 90, found: 1))
        }
    }

    /// With no joint angles the scratch position equals the state's xpos at the initial pose; with
    /// a joint angle set, the state's xpos is unchanged by the query.
    func testScratchKinematicsLeavesTheStateUnchanged() throws {
        let physics = try loadedPhysics()
        let body = 45
        let stateXpos = Array(physics.xpos[(body * 3) ..< (body * 3 + 3)])
        let scratch = try physics.scratchBodyPosition(ofBody: body, withJointAngles: [:])
        XCTAssertEqual(scratch.count, 3)
        for axis in 0 ..< 3 {
            XCTAssertEqual(scratch[axis], stateXpos[axis], accuracy: 1e-12)
        }
        let tables = try physics.modelTables()
        let hinge = try XCTUnwrap(tables.jointType.firstIndex(of: 3), "the scene has a hinge joint")
        _ = try physics.scratchBodyPosition(ofBody: body, withJointAngles: [hinge: 0.3])
        XCTAssertEqual(Array(physics.xpos[(body * 3) ..< (body * 3 + 3)]), stateXpos)
        XCTAssertThrowsError(try physics.scratchBodyPosition(ofBody: 76, withJointAngles: [:])) { error in
            XCTAssertEqual(error as? PhysicsError, .idOutOfRange(kind: .body, id: 76, count: 76))
        }
    }
}
