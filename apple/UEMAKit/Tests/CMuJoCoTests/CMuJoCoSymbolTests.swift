// CMuJoCoSymbolTests.swift: the acceptance test of work item WI-19
// (docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html, row WI-19: "First build: a
// test's mj_versionString() equals \"3.3.0\".") and the Swift call of every symbol the
// row names: mj_loadXML, mj_makeData, mj_step, mj_rnePostConstraint, mj_contactForce,
// mjv_makeScene, mjv_updateScene and the mjData fields qpos, qvel, sensordata, ctrl,
// xpos, xquat, actuator_force, ncon and contact. These calls are the executable example
// for the Swift layer of work item WI-25 (Sources/UEMAPhysics), since no .swift file may
// sit in Sources/CMuJoCo (see shim.h, "WHY NO SWIFT FILE IN THIS FOLDER").
//
// Nothing here was compiled: the machine that wrote it has no Swift toolchain (plan
// section 3, fact 1). The checks that ran on that machine (Windows 11, November 4, 2026):
//   clang 20.1.8 (Visual Studio 18, VC/Tools/Llvm/x64/bin/clang.exe) with
//   -target arm64-apple-macosx14.0 -fsyntax-only, the 14 headers of MuJoCo tag 3.3.0
//   staged as apple/scripts/build_mujoco_xcframework.sh stages them (Headers/mujoco/*.h
//   and Headers/module.modulemap declaring `module mujoco`), and four libc header stubs
//   (stdlib.h, stdio.h, string.h, math.h) in place of the macOS SDK:
//   (a) textual: a C file that includes shim.h and calls every function above with the
//       prototype mujoco.h gives it parsed with exit code 0 under -Wall;
//   (b) as a clang module: an Objective-C file holding `@import CMuJoCo;` with
//       -fmodules -fimplicit-module-maps, the include path holding Sources/CMuJoCo, called
//       mj_versionString, mjv_defaultScene and mjVERSION_HEADER and parsed with exit
//       code 0 and one warning, "umbrella header for module 'mujoco' does not include
//       header 'mjxmacro.h'" (MuJoCo's own dist/module.modulemap has the same umbrella).
//
// THE REFERENCE NUMBERS. Measured with Python mujoco 3.3.0 (Python 3.13.3, Windows 11
// x86-64) on November 4, 2026 from the scene in `sceneXML` below, by this procedure:
//     m = mujoco.MjModel.from_xml_path("cmujoco_test_scene.xml"); d = mujoco.MjData(m)
//     mujoco.mj_forward(m, d)                      # the "after forward" numbers
//     for _ in range(400):                         # 400 x 0.005 s = 2.0 s simulated
//         d.ctrl[0] = 0.5; mujoco.mj_step(m, d); mujoco.mj_rnePostConstraint(m, d)
//     f = numpy.zeros(6); mujoco.mj_contactForce(m, d, 0, f)
//     scn = mujoco.MjvScene(m, maxgeom=1000)
//     mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, mujoco.MjvCamera(),
//                            mujoco.mjtCatBit.mjCAT_ALL, scn)
// The same 400 steps without mj_rnePostConstraint gave the same qpos to the last digit
// and cfrc_ext all zeros for the ball; one mj_rnePostConstraint call then gave 41.0920319089545
// in the ball's force-z slot. The numbers are copied below with 17 significant digits.
//
// TOLERANCE. 1e-6 on every position, velocity, sensor and force: the line of
// specification REQ-B-1 ("every qpos entry equals Python mujoco 3.3.0's within 1e-6, the
// observer recorder's position quantum"). The reference ran on x86-64 and this test runs
// on arm64, where the C compiler may contract a*b+c into one fused multiply-add, so
// bit equality is not expected; integer counts and exact constants are compared exactly.
//
// Swift's view of the C declarations (Swift's C importer rules): mjtNum is Double;
// int is Int32; a returned pointer is an implicitly unwrapped Optional; `mjtNum*` is
// UnsafeMutablePointer<Double>; a fixed C array such as mjContact.geom[2] is a tuple;
// the C enums mjtCatBit and mjtGeom are structs with a rawValue; the integer macro
// mjVERSION_HEADER is an Int32 constant; an imported C struct such as mjvScene has a
// zero-filling init().
//
// FIRST BUILD ON A MAC (plan section 8; this file is step 4 of shim.h's checklist)
//  [ ] 1. bash apple/scripts/build_mujoco_xcframework.sh : exit code 0 (WI-18).
//  [ ] 2. cd apple/UEMAKit && swift package resolve : exit code 0 (WI-17).
//  [ ] 3. xcodebuild -scheme UEMAKit test -destination 'platform=macOS'
//         -only-testing:CMuJoCoTests : every test of this file passes;
//         testVersionStringIs330 is the acceptance line of WI-19.
//  [ ] 4. If a comparison fails, copy the failure lines (name, index, both numbers)
//         into apple/FIRST_BUILD_RECORD.md (WI-41). A difference above 1e-6 in qpos is
//         the REQ-B-1 question of arm64 against x86-64 arithmetic and goes to that
//         record, not into a wider tolerance here.
//  [ ] 5. If compilation fails on `mjvScene()`, `mjCAT_ALL.rawValue` or a tuple field,
//         record the compiler's message: those three are the importer rules above that
//         no compiler checked here.

import CMuJoCo
import Foundation
import XCTest

/// Calls every MuJoCo symbol of work item WI-19 from Swift on a two-body test scene and
/// compares the results with Python mujoco 3.3.0.
final class CMuJoCoSymbolTests: XCTestCase {
    /// The test scene: a hinge arm (one capsule, 0.4105014400690663 kg) driven by one
    /// motor with ctrlrange -1 to 1, a free ball (radius 0.1 m, 4.188790204786391 kg)
    /// 0.5 m above a plane, two joint sensors, timestep 0.005 s. The same bytes as the
    /// file the reference numbers were measured from.
    static let sceneXML = """
    <mujoco model="cmujoco_test_scene">
      <option timestep="0.005" gravity="0 0 -9.81"/>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 0.1"/>
        <body name="arm" pos="0 0 1">
          <joint name="hinge" type="hinge" axis="0 1 0" damping="0.5"/>
          <geom name="rod" type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
        </body>
        <body name="ball" pos="0.5 0 0.5">
          <freejoint name="ball_free"/>
          <geom name="ball_geom" type="sphere" size="0.1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="hinge_motor" joint="hinge" ctrllimited="true" ctrlrange="-1 1" gear="1"/>
      </actuator>
      <sensor>
        <jointpos name="hinge_pos" joint="hinge"/>
        <jointvel name="hinge_vel" joint="hinge"/>
      </sensor>
    </mujoco>
    """

    /// REQ-B-1's equality line for positions, used for every floating-point comparison.
    static let tolerance = 1e-6

    /// Physics steps run before the state comparison: 400 x 0.005 s = 2.0 s simulated.
    static let stepCount = 400

    /// The control written to ctrl[0] before every physics step (motor torque 0.5 N m,
    /// gear 1).
    static let control = 0.5

    /// The ball's weight in newtons, 4.188790204786391 kg x 9.81 m/s^2, as Python
    /// computed it.
    static let ballWeightNewtons = 41.0920319089545

    /// A loaded model and its data; `release()` frees both.
    struct Loaded {
        /// The compiled model returned by mj_loadXML.
        let model: UnsafeMutablePointer<mjModel>
        /// The simulation data returned by mj_makeData.
        let data: UnsafeMutablePointer<mjData>

        /// Frees the data, then the model.
        func release() {
            mj_deleteData(data)
            mj_deleteModel(model)
        }
    }

    // MARK: - Helpers

    /// The text of a NUL-terminated byte buffer, up to its first zero byte.
    func text(of buffer: [CChar]) -> String {
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// Writes `sceneXML` into a new folder under the temporary directory and returns the
    /// file's URL; the folder is removed when the test ends.
    func writeScene() throws -> URL {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("CMuJoCoSymbolTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: folder)
        }
        let file = folder.appendingPathComponent("cmujoco_test_scene.xml")
        try Self.sceneXML.write(to: file, atomically: true, encoding: .utf8)
        return file
    }

    /// Calls mj_loadXML on `file` with no virtual file system and a 1,000-byte error
    /// buffer; returns the model (nil on failure) and the buffer's text.
    func loadXML(_ file: URL) -> (model: UnsafeMutablePointer<mjModel>?, error: String) {
        var errorBuffer = [CChar](repeating: 0, count: 1000)
        let errorSize = Int32(errorBuffer.count)
        let model: UnsafeMutablePointer<mjModel>? = file.withUnsafeFileSystemRepresentation { path in
            mj_loadXML(path, nil, &errorBuffer, errorSize)
        }
        return (model, text(of: errorBuffer))
    }

    /// Loads the test scene and makes its data.
    func loadScene() throws -> Loaded {
        let result = loadXML(try writeScene())
        let model = try XCTUnwrap(result.model, "mj_loadXML returned NULL: \(result.error)")
        let data = try XCTUnwrap(mj_makeData(model), "mj_makeData returned NULL")
        return Loaded(model: model, data: data)
    }

    /// `count` doubles starting at `pointer`, copied into an array.
    func values(_ pointer: UnsafeMutablePointer<mjtNum>, count: Int) -> [Double] {
        Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    /// Fails unless `actual` and `expected` have the same length and each pair of
    /// entries differs by at most `tolerance`.
    func assertClose(_ actual: [Double], _ expected: [Double], _ name: String,
                     file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, "\(name): entry count", file: file, line: line)
        for (index, pair) in zip(actual, expected).enumerated() {
            XCTAssertEqual(pair.0, pair.1, accuracy: Self.tolerance, "\(name)[\(index)]",
                           file: file, line: line)
        }
    }

    /// Runs `stepCount` physics steps, writing `control` to ctrl[0] before each and
    /// calling mj_rnePostConstraint after each when `withRNE` is true.
    func run(_ scene: Loaded, withRNE: Bool) {
        for _ in 0..<Self.stepCount {
            scene.data.pointee.ctrl[0] = Self.control
            mj_step(scene.model, scene.data)
            if withRNE {
                mj_rnePostConstraint(scene.model, scene.data)
            }
        }
    }

    // MARK: - Version

    /// The acceptance line of WI-19: the linked library reports version 3.3.0, and the
    /// header the module imported is the 3.3.0 header (mujoco.h: mjVERSION_HEADER 330).
    func testVersionStringIs330() throws {
        let version = try XCTUnwrap(mj_versionString(), "mj_versionString returned NULL")
        XCTAssertEqual(String(cString: version), "3.3.0")
        XCTAssertEqual(mj_version(), 330)
        XCTAssertEqual(mjVERSION_HEADER, 330)
    }

    // MARK: - Loading

    /// mj_loadXML compiles the scene: the model sizes equal Python's (nq 8: 1 hinge + 7
    /// free joint; nv 7; nu 1; nbody 3 with the world; ngeom 3; nsensordata 2).
    func testLoadXMLGivesPythonModelSizes() throws {
        let scene = try loadScene()
        defer { scene.release() }
        // Fields are read through the pointer, never by copying the whole struct.
        let model = scene.model
        XCTAssertEqual(model.pointee.nq, 8)
        XCTAssertEqual(model.pointee.nv, 7)
        XCTAssertEqual(model.pointee.nu, 1)
        XCTAssertEqual(model.pointee.nbody, 3)
        XCTAssertEqual(model.pointee.ngeom, 3)
        XCTAssertEqual(model.pointee.nsensordata, 2)
        XCTAssertEqual(model.pointee.opt.timestep, 0.005)
    }

    /// mj_loadXML on a missing file returns NULL and writes a message into the buffer.
    func testLoadXMLReportsMissingFile() throws {
        let missing = try writeScene().deletingLastPathComponent()
            .appendingPathComponent("no_such_scene.xml")
        let result = loadXML(missing)
        XCTAssertNil(result.model)
        XCTAssertFalse(result.error.isEmpty, "the error buffer holds no message")
        if let model = result.model {
            mj_deleteModel(model)
        }
    }

    // MARK: - State fields

    /// mj_makeData sets qpos to the model's qpos0 (hinge 0 rad; the ball's free joint
    /// at 0.5, 0, 0.5 m with the identity quaternion 1, 0, 0, 0) and leaves xpos and
    /// xquat at zeros; mj_forward fills the body frames (ball at 0.5, 0, 0.5 m; every
    /// xquat the identity). Python mujoco 3.3.0 gave the same arrays from
    /// mujoco.MjData(m) and after mujoco.mj_forward(m, d) on November 4, 2026.
    func testForwardFillsBodyFrames() throws {
        let scene = try loadScene()
        defer { scene.release() }
        assertClose(values(scene.data.pointee.qpos, count: 8),
                    [0, 0.5, 0, 0.5, 1, 0, 0, 0], "qpos after mj_makeData")
        assertClose(values(scene.data.pointee.xpos, count: 9), Array(repeating: 0, count: 9),
                    "xpos before mj_forward")
        assertClose(values(scene.data.pointee.xquat, count: 12), Array(repeating: 0, count: 12),
                    "xquat before mj_forward")
        mj_forward(scene.model, scene.data)
        assertClose(values(scene.data.pointee.xpos, count: 9),
                    [0, 0, 0, 0, 0, 1, 0.5, 0, 0.5], "xpos after mj_forward")
        assertClose(values(scene.data.pointee.xquat, count: 12),
                    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], "xquat after mj_forward")
    }

    /// 400 x (ctrl[0] = 0.5; mj_step; mj_rnePostConstraint): time, qpos, qvel, ctrl,
    /// sensordata, actuator_force, xpos and xquat equal Python's within 1e-6.
    func testStepMatchesPython() throws {
        let scene = try loadScene()
        defer { scene.release() }
        run(scene, withRNE: true)
        // mjData holds fixed arrays of solver statistics, so it is read through the
        // pointer rather than copied.
        let data = scene.data
        XCTAssertEqual(data.pointee.time, 1.9999999999999793, accuracy: 1e-9)
        assertClose(values(data.pointee.qpos, count: 8),
                    [2.191879793718918, 0.5, 0.0, 0.09963281815753985, 1.0, 0.0, 0.0, 0.0],
                    "qpos")
        assertClose(values(data.pointee.qvel, count: 7),
                    [0.30659927259658626, 0.0, 0.0, -4.92742857598558e-16, 0.0, 0.0, 0.0],
                    "qvel")
        assertClose(values(data.pointee.ctrl, count: 1), [Self.control], "ctrl")
        // The joint sensors are evaluated inside mj_step before the integration, so they
        // read the state of one physics step earlier than qpos and qvel above.
        assertClose(values(data.pointee.sensordata, count: 2),
                    [2.190346797355935, 0.30816641359524877], "sensordata")
        assertClose(values(data.pointee.actuator_force, count: 1), [0.5], "actuator_force")
        assertClose(values(data.pointee.xpos, count: 9),
                    [0, 0, 0, 0, 0, 1, 0.5, 0, 0.09963281815753985], "xpos")
        assertClose(values(data.pointee.xquat, count: 12),
                    [1, 0, 0, 0,
                     0.4578923238509339, 0, 0.8890076601235737, 0,
                     1, 0, 0, 0], "xquat")
    }

    // MARK: - Contacts

    /// After the 400 steps the ball rests on the floor: ncon 1, contact geoms (floor 0,
    /// ball 2), penetration -0.0003671818424601514 m, normal (0, 0, 1), and
    /// mj_contactForce's normal force equals the ball's weight, 41.0920319089545 N.
    func testContactForceIsTheBallWeight() throws {
        let scene = try loadScene()
        defer { scene.release() }
        run(scene, withRNE: true)
        XCTAssertEqual(scene.data.pointee.ncon, 1)
        guard scene.data.pointee.ncon >= 1 else {
            return
        }
        let contact = scene.data.pointee.contact[0]
        XCTAssertEqual(contact.geom.0, 0)
        XCTAssertEqual(contact.geom.1, 2)
        XCTAssertEqual(contact.dist, -0.0003671818424601514, accuracy: Self.tolerance)
        assertClose([contact.pos.0, contact.pos.1, contact.pos.2],
                    [0.5, 0.0, -0.0001835909212300757], "contact.pos")
        assertClose([contact.frame.0, contact.frame.1, contact.frame.2], [0, 0, 1],
                    "contact normal")
        var force = [Double](repeating: 0, count: 6)
        mj_contactForce(scene.model, scene.data, 0, &force)
        assertClose(force, [Self.ballWeightNewtons, 0, 0, 0, 0, 0], "mj_contactForce")
    }

    /// mj_rnePostConstraint fills cfrc_ext (nbody x 6: torque 3, then force 3) and
    /// changes no position: without it the ball's row is zeros; one call puts the
    /// ball's weight in its force-z slot (index 2 x 6 + 5 = 17), and qpos is unchanged.
    func testRNEPostConstraintFillsExternalForce() throws {
        let scene = try loadScene()
        defer { scene.release() }
        run(scene, withRNE: false)
        assertClose(Array(values(scene.data.pointee.cfrc_ext, count: 18).suffix(6)),
                    Array(repeating: 0, count: 6), "ball cfrc_ext before mj_rnePostConstraint")
        let qposBefore = values(scene.data.pointee.qpos, count: 8)
        mj_rnePostConstraint(scene.model, scene.data)
        assertClose(Array(values(scene.data.pointee.cfrc_ext, count: 18).suffix(6)),
                    [0, 0, 0, 0, 0, Self.ballWeightNewtons],
                    "ball cfrc_ext after mj_rnePostConstraint")
        XCTAssertEqual(values(scene.data.pointee.qpos, count: 8), qposBefore)
        assertClose(qposBefore,
                    [2.191879793718918, 0.5, 0.0, 0.09963281815753985, 1.0, 0.0, 0.0, 0.0],
                    "qpos without mj_rnePostConstraint")
    }

    // MARK: - Abstract scene

    /// mjv_makeScene with maxgeom 1000 and mjv_updateScene with mjCAT_ALL (7) give the
    /// three geoms in model order: plane (mjtGeom 0), capsule (3), sphere (2). No OpenGL
    /// function is called.
    func testUpdateSceneListsThreeGeoms() throws {
        let scene = try loadScene()
        defer { scene.release() }
        run(scene, withRNE: true)
        var abstractScene = mjvScene()
        var option = mjvOption()
        var camera = mjvCamera()
        mjv_defaultScene(&abstractScene)
        mjv_makeScene(scene.model, &abstractScene, 1000)
        defer { mjv_freeScene(&abstractScene) }
        mjv_defaultOption(&option)
        mjv_defaultCamera(&camera)
        let categories = Int32(truncatingIfNeeded: mjCAT_ALL.rawValue)
        XCTAssertEqual(categories, 7)
        mjv_updateScene(scene.model, scene.data, &option, nil, &camera, categories, &abstractScene)
        XCTAssertEqual(abstractScene.maxgeom, 1000)
        XCTAssertEqual(abstractScene.ngeom, 3)
        let count = Int(max(0, min(abstractScene.ngeom, 3)))
        let types = (0..<count).map { Int(abstractScene.geoms[$0].type) }
        XCTAssertEqual(types, [Int(mjGEOM_PLANE.rawValue), Int(mjGEOM_CAPSULE.rawValue),
                               Int(mjGEOM_SPHERE.rawValue)])
        XCTAssertEqual(types, [0, 3, 2])
    }
}
