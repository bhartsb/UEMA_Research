// MuJoCoPhysics.swift: the MuJoCo physics of the UEMA Apple port (work item WI-25, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification
// docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html REQ-B-3). The other file of the item is
// PhysicsState.swift (errors, the state copy, the model tables and three small protocols).
//
// WHAT THIS FILE DOES. One final class, MuJoCoPhysics, owns one mjModel and one mjData of
// MuJoCo 3.3.0 (through `import CMuJoCo`, work item WI-19) and implements UEMACore's Physics
// (with PhysicsReading) and SceneGeomReading, plus this module's PhysicsModelDescribing,
// ActuatorParameterWriting and ScratchKinematicsQuerying.
//   INPUT   load: the URL of the bundle's scene.xml (UEMAScene's ResourceSceneBundle.sceneURL,
//           WI-26), opened by mj_loadXML. write: nu controls (90 on the released scene; see
//           THE MOTOR COUNT below).
//   OPERATION one physics step = mj_step then mj_rnePostConstraint (mimo_env.py:573-575,
//           `_single_mujoco_step`), 0.005 s; one env step = write the controls, then 2 physics
//           steps (mimo_env.py:608-619 `do_simulation` with `frame_skip=2`, env.py:1157); one model
//           step = 5 env steps = 10 physics steps = 0.050 s (UEMACore ModelStep constants; the
//           Physics protocol's default stepEnvStep and stepModelStep in Protocols.swift:472-490
//           run exactly this loop and are used unchanged).
//   OUTPUT  flat copies of qpos, qvel, ctrl, sensordata, xpos, xquat, xmat, xaxis, xanchor,
//           actuator_force, the contacts (ncon of them, each with mj_contactForce's 6 numbers),
//           a body's 6D velocity (mj_objectVelocity), the scene's geoms (mjv_updateScene) and the
//           eye cameras' poses (cam_xpos, cam_xmat).
//
// THE CONTROL CLIP (the plan row: "writes ctrl clipped to -1..1"). MIMo's actuation model,
// SpringDamperModel (the default, mimo_env.py:322), clips every action to its action space and
// writes data.ctrl (actuation.py:153-154: `self.control_input = np.clip(action,
// self.action_space.low, self.action_space.high)`, `self.env.data.ctrl[self.actuators] =
// self.control_input`). That action space is actuator_forcerange as compiled
// (actuation.py:136-138), built once at env construction (mimo_env.py:447), so a later
// modulation of actuator_forcerange (env.py:2845) does not move it. On the released scene the
// control range is -1 to 1 for all 90 motors, while the force range is narrower for some: the
// low end lies above -1 for 42 motors and the high end below 1 for 14 (Python mujoco 3.3.0 on
// the bundle's scene.xml, November 4, 2026); env.py:2916-2919 records the measured effect ("18
// of 90 actuators ... settle to a different value than they were given"). This class therefore
// clips each control to [max(-1, forcerange low), min(1, forcerange high)], the bounds read at
// load, each bound rounded through float32 as MIMo's action space stores it (actuation.py:136,
// `.astype(np.float32)`; 0.3 becomes 0.30000001192092896): the plan row's -1..1 and the Python
// clip at once. MIMo's per-physics-step hook
// `actuation_model.substep_update()` (mimo_env.py:617) is the base class's `pass` for
// SpringDamperModel (actuation.py:76-81; SpringDamperModel, actuation.py:107-197, does not
// override it), so a physics step here runs nothing before mj_step.
//
// THE MOTOR COUNT. The released scene compiles to nu = 90: bundle.json model_counts.nu 90, its
// observer_header.nu 90 (the trainer's own run), Python mujoco 3.3.0 on scene.xml 90 (November 4,
// 2026), env.py:2918 "18 of 90 actuators". UEMACore's MotorControls.count is 90 (it read 92
// before November 4, 2026, a figure from another body file), so `write(controls: MotorControls)`
// writes on the released scene and throws PhysicsError.wrongCount only for a scene of another nu;
// `write(controls: [Double])` takes nu numbers for any scene.
//
// MEMORY. Python measured one mjData arena of this scene at 380 MB (plan row WI-40). This class
// holds one mjData for the state and, only after the first scratchBodyPosition call, a second one
// for the scratch kinematics (reused across calls; mj_resetData before each query gives the state
// mj_makeData gives, qpos = qpos0). The abstract scene of sceneGeoms is allocated at its first
// call with `maxSceneGeoms` geoms (10,000 by default: mujoco/renderer.py:35, `max_geom: int =
// 10000`, of the Python package the trainer renders with).
//
// SOURCES READ BEFORE WRITING (November 4, 2026). No web page was fetched for this file: the C
// prototypes and struct fields were read in the include/mujoco headers of the mujoco 3.3.0
// Python wheel, the same tag WI-18 builds:
//   mujoco.h:99   mjModel* mj_loadXML(const char* filename, const mjVFS* vfs, char* error, int error_sz);
//   mujoco.h:131  void mj_step(const mjModel* m, mjData* d);
//   mujoco.h:140  void mj_forward(const mjModel* m, mjData* d);
//   mujoco.h:177  void mj_deleteModel(mjModel* m);
//   mujoco.h:184  mjData* mj_makeData(const mjModel* m);
//   mujoco.h:191  void mj_resetData(const mjModel* m, mjData* d);
//   mujoco.h:221  void mj_deleteData(mjData* d);
//   mujoco.h:387  void mj_rnePostConstraint(const mjModel* m, mjData* d);
//   mujoco.h:473  int mj_name2id(const mjModel* m, int type, const char* name);
//   mujoco.h:476  const char* mj_id2name(const mjModel* m, int type, int id);  ("returns NULL if name not found")
//   mujoco.h:496  void mj_objectVelocity(const mjModel* m, const mjData* d, int objtype, int objid, mjtNum res[6], int flg_local);
//   mujoco.h:508  void mj_contactForce(const mjModel* m, const mjData* d, int id, mjtNum result[6]);
//   mujoco.h:591, 667, 683, 686, 689, 692  mjv_defaultCamera, mjv_defaultOption, mjv_defaultScene,
//                 mjv_makeScene(m, scn, int maxgeom), mjv_freeScene(scn),
//                 mjv_updateScene(m, d, const mjvOption*, const mjvPerturb*, mjvCamera*, int catmask, mjvScene*)
//   mjdata.h      mjContact: mjtNum dist; mjtNum pos[3]; mjtNum frame[9]; int geom[2] ("geom ids; -1 for
//                 flex"); mjData: int ncon; mjtNum time; mjtNum* qpos, qvel, ctrl, sensordata, xpos,
//                 xquat, xmat, xanchor, xaxis, cam_xpos, cam_xmat, actuator_force; mjContact* contact.
//   mjmodel.h     int nq, nv, nu, nbody, njnt, ngeom, ncam, nsensor, nsensordata; mjOption opt
//                 (mjtNum timestep); mjtNum* qpos0, body_mass, jnt_range (njnt x 2), actuator_gainprm
//                 (nu x mjNGAIN), actuator_ctrlrange, actuator_forcerange (nu x 2), actuator_gear
//                 (nu x 6); int* body_parentid, body_rootid, geom_bodyid (ngeom), jnt_type,
//                 jnt_qposadr, jnt_dofadr, jnt_bodyid,
//                 actuator_trntype, actuator_gaintype, actuator_biastype, actuator_trnid (nu x 2),
//                 sensor_dim, sensor_adr; mjtByte* jnt_limited, actuator_ctrllimited,
//                 actuator_forcelimited; mjmodel.h:39 `#define mjNGAIN 10`.
//   mjvisualize.h mjvGeom: int type; float size[3], pos[3], mat[9], rgba[4]; mjvScene: int maxgeom,
//                 ngeom; mjvGeom* geoms; mjtCatBit mjCAT_ALL = 7.
//   Enum values checked with Python mujoco 3.3.0: mjOBJ_BODY 1, mjOBJ_JOINT 3, mjOBJ_GEOM 5,
//   mjOBJ_CAMERA 7, mjOBJ_ACTUATOR 19, mjOBJ_SENSOR 20, mjCAT_ALL 7. They are written as numbers
//   below so that this file does not depend on how Swift imports a C enum.
// How Swift's C importer gives these (shim.h header, WI-19): mjtNum is Double, int is Int32,
// mjtByte is UInt8, a pointer field is an implicitly unwrapped pointer, a fixed C array a tuple.
//
// MEASURED ON THE RELEASED SCENE (Python mujoco 3.3.0, bundle scene.xml, November 4, 2026), the
// numbers the tests compare against: nq 146, nv 136, nu 90, nbody 76, njnt 104, nsensor 56,
// nsensordata 168, ngeom 188, ncam 6, timestep 0.005 s; after mj_forward and 200 physics steps of
// zero control, time 1.0000000000000007 s, no NaN in qpos, ncon 28; mjv_updateScene with default
// option and catmask 7 gives 182 geoms; camera ids watch_cam 0, eye_left 1, fovea_left 2,
// ring_left 3, eye_right 4, fovea_right 5; body left_eye id 45.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). No function of MuJoCo's classic OpenGL renderer is named
// (CMuJoCoRendererFreeTests.swift scans for its prefix).
//
// FIRST BUILD ON A MAC (this work item's acceptance: "loads the WI-15 bundle, steps 20 model
// steps (1.0 s); nq equals 146; simulated time advanced 1.0 s; no NaN in qpos").
//  [ ] 1. Plan section 8 steps 1 to 4 first: `bash apple/scripts/build_mujoco_xcframework.sh`
//         exit 0 (WI-18), `cd apple/UEMAKit && swift package resolve` exit 0 (WI-17).
//  [ ] 2. `xcodebuild -scheme UEMAKit-Package build -destination 'platform=macOS'` : exit 0 with
//         UEMAPhysics compiled. Constructs to check first if it fails: `mjvScene()`,
//         `mjvOption()`, `mjvCamera()` (the zero initializers Swift gives an imported C struct);
//         `&option` passed to a `const mjvOption*` parameter; the String `path` passed to a
//         `const char*` parameter inside the `withUnsafeMutableBufferPointer` closure of load;
//         the tuple fields `contact.geom.0`, `geom.mat.8`; the overload pair write(controls:
//         MotorControls) and write(controls: [Double]); the int* fields geom_bodyid and
//         body_rootid read into PhysicsModelTables.
//  [ ] 3. `xcodebuild -scheme UEMAKit-Package test -destination 'platform=macOS'
//         -only-testing:UEMAPhysicsTests` : every test passes (18 in MuJoCoPhysicsTests, 5 in
//         PhysicsStateTests, 1 in UEMAPhysicsPackageTests). The acceptance test is
//         MuJoCoPhysicsTests.testTwentyModelStepsAdvanceOneSecondWithFiniteQpos (nq 146, time
//         1.0 s within 1e-9 s, 200 physics-step callbacks, every qpos finite). The tests find
//         scene.xml from their own path (#filePath), three folders up, then
//         Sources/UEMAScene/Resources/crib_age24_trunk_v1/scene.xml.
//  [ ] 4. testMotorControlsWriteMatchesTheModelCount passes by writing MotorControls (90 against nu
//         90, THE MOTOR COUNT); it passed on the Linux harness of apple/FIRST_BUILD_CHECKLIST.md.
//  [ ] 5. Write each command, exit code and clock time (AM/PM Pacific) into
//         apple/FIRST_BUILD_RECORD.md (WI-41); the plan page's WI-25 row moves to VERIFIED or
//         FAILED with the commit hash.

import CMuJoCo
import Foundation
import UEMACore

/// The MuJoCo physics of one loaded scene: one mjModel and one mjData, stepped as UEMACore's
/// Physics protocol defines (one physics step = mj_step + mj_rnePostConstraint, 0.005 s; one env
/// step = write + 2 physics steps; one model step = 5 env steps, 0.050 s) and read through
/// PhysicsReading, SceneGeomReading and PhysicsModelDescribing. Not thread-safe: one caller
/// drives it. Injected by reference into the sense readers and the command decoder.
public final class MuJoCoPhysics: Physics, SceneGeomReading, PhysicsModelDescribing,
    ActuatorParameterWriting, ScratchKinematicsQuerying
{
    /// The abstract scene's geom buffer by default: 10,000 (mujoco/renderer.py:35).
    public static let defaultMaxSceneGeoms = 10_000

    /// The byte size of the error buffer handed to mj_loadXML: 1,000 (the size MuJoCo's own
    /// sample programs use for `char error[1000]`).
    public static let loadErrorBufferBytes = 1000

    /// mjNGAIN, the gain parameters per actuator: 10 (mjmodel.h:39).
    static let gainParametersPerActuator = 10

    /// mjCAT_ALL, static, dynamic and decorative geoms: 7 (mjvisualize.h, mjtCatBit).
    static let allGeomCategories: Int32 = 7

    /// The largest difference allowed between the scene's timestep and 0.005 s, in seconds.
    static let timestepTolerance = 1e-12

    /// The geom buffer size of the abstract scene sceneGeoms() fills.
    public let maxSceneGeoms: Int

    /// The scene file of the current load; nil before a load.
    public private(set) var sceneURL: URL?

    /// The low end of each motor's control clip, nu numbers: max(-1, actuator_forcerange low
    /// rounded through Float) as compiled (see THE CONTROL CLIP in this file's header). Empty before a load.
    public private(set) var controlLowerBounds: [Double] = []

    /// The high end of each motor's control clip: min(1, actuator_forcerange high rounded through
    /// Float) as compiled.
    public private(set) var controlUpperBounds: [Double] = []

    /// Physics steps run since the last load or reset.
    public private(set) var physicsStepCount = 0

    private var model: UnsafeMutablePointer<mjModel>?
    private var data: UnsafeMutablePointer<mjData>?
    private var scratchData: UnsafeMutablePointer<mjData>?
    private var abstractScene: UnsafeMutablePointer<mjvScene>?

    /// A physics with no scene loaded; `maxSceneGeoms` must be 1 or more.
    public init(maxSceneGeoms: Int = MuJoCoPhysics.defaultMaxSceneGeoms) {
        precondition(maxSceneGeoms > 0, "the abstract scene needs at least 1 geom; got \(maxSceneGeoms)")
        self.maxSceneGeoms = maxSceneGeoms
    }

    deinit {
        release()
    }

    // MARK: Loading

    /// True once `load(sceneXML:)` succeeded.
    public var isLoaded: Bool { model != nil && data != nil }

    /// Opens `url` with mj_loadXML (no virtual file system: the include and texture files are
    /// read from the scene's folder), checks the timestep is 0.005 s, makes the data, reads the
    /// control clip bounds and runs mj_forward so that every position array holds the initial
    /// pose before the first step. A previous load is released first.
    public func load(sceneXML url: URL) throws {
        let path = url.path
        guard FileManager.default.fileExists(atPath: path) else {
            throw PhysicsError.sceneFileMissing(path: path)
        }
        release()
        var errorBuffer = [CChar](repeating: 0, count: MuJoCoPhysics.loadErrorBufferBytes)
        let loadedModel: UnsafeMutablePointer<mjModel>? = errorBuffer.withUnsafeMutableBufferPointer { buffer in
            mj_loadXML(path, nil, buffer.baseAddress, Int32(buffer.count))
        }
        guard let newModel = loadedModel else {
            throw PhysicsError.loadFailed(path: path, message: CArrays.text(errorBuffer))
        }
        let timestep = newModel.pointee.opt.timestep
        guard abs(timestep - ModelStep.physicsTimestepSeconds) <= MuJoCoPhysics.timestepTolerance else {
            mj_deleteModel(newModel)
            throw PhysicsError.timestepMismatch(
                expectedSeconds: ModelStep.physicsTimestepSeconds, foundSeconds: timestep)
        }
        guard let newData = mj_makeData(newModel) else {
            mj_deleteModel(newModel)
            throw PhysicsError.dataAllocationFailed
        }
        let nu = Int(newModel.pointee.nu)
        let forceLow = CArrays.column(newModel.pointee.actuator_forcerange, rows: nu, width: 2, index: 0)
        let forceHigh = CArrays.column(newModel.pointee.actuator_forcerange, rows: nu, width: 2, index: 1)
        // MIMo's action space holds the force range as float32 (actuation.py:136,
        // `.astype(np.float32)`), so each bound is rounded through Float as the Python clip uses it.
        controlLowerBounds = forceLow.map { max(Double(Float($0)), MotorControls.range.lowerBound) }
        controlUpperBounds = forceHigh.map { min(Double(Float($0)), MotorControls.range.upperBound) }
        mj_forward(newModel, newData)
        model = newModel
        data = newData
        sceneURL = url
        physicsStepCount = 0
    }

    /// Resets the state to the model's initial state (mj_resetData: qpos = qpos0, qvel = 0,
    /// ctrl = 0, time = 0) and runs mj_forward.
    public func reset() throws {
        let (m, d) = try loadedPointers()
        mj_resetData(m, d)
        mj_forward(m, d)
        physicsStepCount = 0
    }

    /// Sets qpos (nq numbers) and qvel (nv numbers) and runs mj_forward, as gymnasium's
    /// MujocoEnv.set_state does for the Python env.
    public func setState(qpos: [Double], qvel: [Double]) throws {
        let (m, d) = try loadedPointers()
        let nq = Int(m.pointee.nq)
        let nv = Int(m.pointee.nv)
        try CArrays.check(qpos, name: "qpos", count: nq)
        try CArrays.check(qvel, name: "qvel", count: nv)
        CArrays.write(qpos, into: d.pointee.qpos)
        CArrays.write(qvel, into: d.pointee.qvel)
        mj_forward(m, d)
    }

    // MARK: Writing and stepping

    /// Writes the 90 controls of UEMACore's MotorControls. Throws PhysicsError.wrongCount when the
    /// model's nu is not MotorControls.count (THE MOTOR COUNT in this file's header).
    public func write(controls: MotorControls) throws {
        try write(controls: controls.values)
    }

    /// Writes nu controls into mjData.ctrl, each clipped to [controlLowerBounds[i],
    /// controlUpperBounds[i]] (actuation.py:153-154 within -1..1). A count other than nu throws
    /// PhysicsError.wrongCount; a NaN or infinite control throws PhysicsError.nonFinite.
    public func write(controls: [Double]) throws {
        let (m, d) = try loadedPointers()
        let nu = Int(m.pointee.nu)
        try CArrays.check(controls, name: "controls", count: nu)
        // A model without actuators (nu 0) has nothing to write; its ctrl pointer is not read.
        guard nu > 0 else { return }
        guard let ctrl = d.pointee.ctrl else { throw PhysicsError.notLoaded }
        for index in 0 ..< nu {
            ctrl[index] = min(max(controls[index], controlLowerBounds[index]), controlUpperBounds[index])
        }
    }

    /// One physics step of 0.005 s: mj_step then mj_rnePostConstraint (mimo_env.py:573-575).
    public func stepPhysicsStep() throws {
        let (m, d) = try loadedPointers()
        mj_step(m, d)
        mj_rnePostConstraint(m, d)
        physicsStepCount += 1
    }

    /// One env step (0.010 s) from nu raw controls: `write(controls:)` then 2 physics steps,
    /// calling `afterEachPhysicsStep` after each one. The nu-count twin of the Physics protocol's
    /// stepEnvStep(controls:afterEachPhysicsStep:).
    public func stepEnvStep(
        controlValues: [Double], afterEachPhysicsStep: (any PhysicsReading) throws -> Void = { _ in }
    ) throws {
        try write(controls: controlValues)
        for _ in 0 ..< ModelStep.physicsStepsPerEnvStep {
            try stepPhysicsStep()
            try afterEachPhysicsStep(self)
        }
    }

    /// One model step (0.050 s): 5 env steps holding the same nu raw controls, 10 physics steps.
    /// The nu-count twin of the Physics protocol's stepModelStep(controls:afterEachPhysicsStep:).
    public func stepModelStep(
        controlValues: [Double], afterEachPhysicsStep: (any PhysicsReading) throws -> Void = { _ in }
    ) throws {
        for _ in 0 ..< ModelStep.envStepsPerModelStep {
            try stepEnvStep(controlValues: controlValues, afterEachPhysicsStep: afterEachPhysicsStep)
        }
    }

    // MARK: PhysicsReading

    /// The sizes of the loaded model; all zero before a load.
    public var sizes: PhysicsSizes {
        guard let m = model else {
            return PhysicsSizes(nq: 0, nv: 0, nu: 0, nbody: 0, njnt: 0, nsensordata: 0, ngeom: 0, ncam: 0)
        }
        return PhysicsSizes(
            nq: Int(m.pointee.nq), nv: Int(m.pointee.nv), nu: Int(m.pointee.nu),
            nbody: Int(m.pointee.nbody), njnt: Int(m.pointee.njnt),
            nsensordata: Int(m.pointee.nsensordata), ngeom: Int(m.pointee.ngeom),
            ncam: Int(m.pointee.ncam))
    }

    /// The physics timestep of the loaded scene in seconds (mjModel.opt.timestep); 0 before a load.
    public var timestep: Double { model?.pointee.opt.timestep ?? 0 }

    /// Simulated time in seconds (mjData.time); 0 before a load.
    public var time: Double { data?.pointee.time ?? 0 }

    /// Generalized positions, nq; empty before a load.
    public var qpos: [Double] { dataArray(count: { $0.nq }, pick: { $0.pointee.qpos }) }

    /// Generalized velocities, nv.
    public var qvel: [Double] { dataArray(count: { $0.nv }, pick: { $0.pointee.qvel }) }

    /// Sensor values, nsensordata.
    public var sensordata: [Double] { dataArray(count: { $0.nsensordata }, pick: { $0.pointee.sensordata }) }

    /// Body positions, nbody x 3, metres.
    public var xpos: [Double] { dataArray(count: { $0.nbody * 3 }, pick: { $0.pointee.xpos }) }

    /// Body orientations, nbody x 4, quaternion w x y z.
    public var xquat: [Double] { dataArray(count: { $0.nbody * 4 }, pick: { $0.pointee.xquat }) }

    /// Body rotation matrices, nbody x 9, row-major.
    public var xmat: [Double] { dataArray(count: { $0.nbody * 9 }, pick: { $0.pointee.xmat }) }

    /// Joint axes in the world frame, njnt x 3.
    public var xaxis: [Double] { dataArray(count: { $0.njnt * 3 }, pick: { $0.pointee.xaxis }) }

    /// Joint anchors in the world frame, njnt x 3, metres.
    public var xanchor: [Double] { dataArray(count: { $0.njnt * 3 }, pick: { $0.pointee.xanchor }) }

    /// Actuator forces in actuation space, nu.
    public var actuatorForce: [Double] { dataArray(count: { $0.nu }, pick: { $0.pointee.actuator_force }) }

    /// The controls in mjData.ctrl, nu (after the clip of `write(controls:)`).
    public var control: [Double] { dataArray(count: { $0.nu }, pick: { $0.pointee.ctrl }) }

    /// The number of contacts of the current physics step (mjData.ncon); 0 before a load.
    public var contactCount: Int { Int(data?.pointee.ncon ?? 0) }

    /// The contacts of the current physics step: geom ids, distance, position, frame from
    /// mjContact and the 6 numbers of mj_contactForce (force 3 then torque 3, newtons and
    /// newton-metres, in the contact frame).
    public var contacts: [Contact] {
        guard let m = model, let d = data, let contactArray = d.pointee.contact else { return [] }
        let ncon = Int(d.pointee.ncon)
        var result: [Contact] = []
        result.reserveCapacity(ncon)
        for index in 0 ..< ncon {
            let c = contactArray[index]
            var force = [Double](repeating: 0, count: 6)
            mj_contactForce(m, d, Int32(index), &force)
            do {
                result.append(try Contact(
                    geomIds: [Int(c.geom.0), Int(c.geom.1)],
                    distance: c.dist,
                    position: [c.pos.0, c.pos.1, c.pos.2],
                    frame: [c.frame.0, c.frame.1, c.frame.2, c.frame.3, c.frame.4,
                            c.frame.5, c.frame.6, c.frame.7, c.frame.8],
                    force: force))
            } catch {
                preconditionFailure("Contact receives 2, 3, 9 and 6 numbers here by construction: \(error)")
            }
        }
        return result
    }

    /// The force and torque of contact `id` (0 ..< ncon) in the contact frame: mj_contactForce's
    /// 6 numbers, the normal force first.
    public func contactForce(_ id: Int) throws -> [Double] {
        let (m, d) = try loadedPointers()
        let ncon = Int(d.pointee.ncon)
        guard (0 ..< ncon).contains(id) else {
            throw PhysicsError.contactOutOfRange(id: id, count: ncon)
        }
        var force = [Double](repeating: 0, count: 6)
        mj_contactForce(m, d, Int32(id), &force)
        return force
    }

    /// The id of the named object of `kind` (mj_name2id); nil when the model has no such name or
    /// no scene is loaded.
    public func id(of kind: MuJoCoObjectKind, named name: String) -> Int? {
        guard let m = model else { return nil }
        let found = mj_name2id(m, MuJoCoPhysics.objectType(kind), name)
        return found >= 0 ? Int(found) : nil
    }

    /// The 6D velocity of body `bodyId` at its origin in the world frame, rotational 3 then
    /// linear 3 (mj_objectVelocity with flg_local 0, proprio_adapter.py:210-211), radians per
    /// second and metres per second. A body id outside 0 ..< nbody, or a call before a load, is
    /// a programming error and stops the program (the protocol method cannot throw).
    public func bodyVelocity(bodyId: Int) -> [Double] {
        guard let m = model, let d = data else {
            preconditionFailure("bodyVelocity(bodyId:) called before load(sceneXML:)")
        }
        let nbody = Int(m.pointee.nbody)
        precondition((0 ..< nbody).contains(bodyId), "body id \(bodyId) outside 0 ..< \(nbody)")
        var velocity = [Double](repeating: 0, count: 6)
        mj_objectVelocity(m, d, MuJoCoPhysics.objectType(.body), Int32(bodyId), &velocity, 0)
        return velocity
    }

    /// A copy of the current state.
    public var state: PhysicsState { PhysicsState(reading: self) }

    // MARK: SceneGeomReading

    /// Every geom of the current state as mjv_updateScene gives it with mjv_defaultOption, a
    /// default free camera, no perturbation and catmask mjCAT_ALL (7), the call the Python
    /// renderer makes (mujoco/renderer.py update_scene). 182 geoms on the released scene at its
    /// initial pose (Python mujoco 3.3.0, November 4, 2026).
    public func sceneGeoms() throws -> [SceneGeom] {
        let (m, d) = try loadedPointers()
        let scene = try sceneBuffer(model: m)
        var option = mjvOption()
        mjv_defaultOption(&option)
        var camera = mjvCamera()
        mjv_defaultCamera(&camera)
        mjv_updateScene(m, d, &option, nil, &camera, MuJoCoPhysics.allGeomCategories, scene)
        guard let buffer = scene.pointee.geoms else { return [] }
        let ngeom = Int(scene.pointee.ngeom)
        var geoms: [SceneGeom] = []
        geoms.reserveCapacity(ngeom)
        for index in 0 ..< ngeom {
            let g = buffer[index]
            geoms.append(try SceneGeom(
                type: Int(g.type),
                size: [g.size.0, g.size.1, g.size.2],
                position: [g.pos.0, g.pos.1, g.pos.2],
                rotation: [g.mat.0, g.mat.1, g.mat.2, g.mat.3, g.mat.4, g.mat.5, g.mat.6, g.mat.7, g.mat.8],
                rgba: [g.rgba.0, g.rgba.1, g.rgba.2, g.rgba.3]))
        }
        return geoms
    }

    /// The current world pose of one eye camera: cam_xpos (3, metres) and cam_xmat (9, row-major)
    /// of the camera named `camera.mujocoName`.
    public func cameraPose(_ camera: EyeCamera) throws -> CameraPose {
        let (_, d) = try loadedPointers()
        guard let cameraId = id(of: .camera, named: camera.mujocoName) else {
            throw PhysicsError.unknownName(kind: .camera, name: camera.mujocoName)
        }
        return try CameraPose(
            position: CArrays.doubles(d.pointee.cam_xpos, offset: cameraId * 3, count: 3),
            rotation: CArrays.doubles(d.pointee.cam_xmat, offset: cameraId * 9, count: 9))
    }

    // MARK: PhysicsModelDescribing

    /// The constants of the loaded model, copied now.
    public func modelTables() throws -> PhysicsModelTables {
        let (m, _) = try loadedPointers()
        let nbody = Int(m.pointee.nbody)
        let njnt = Int(m.pointee.njnt)
        let nu = Int(m.pointee.nu)
        let nsensor = Int(m.pointee.nsensor)
        let ncam = Int(m.pointee.ncam)
        let nq = Int(m.pointee.nq)
        let ngeom = Int(m.pointee.ngeom)
        return try PhysicsModelTables(
            timestep: m.pointee.opt.timestep,
            bodyNames: names(of: .body, count: nbody, model: m),
            bodyParentID: CArrays.ints(m.pointee.body_parentid, count: nbody),
            bodyMass: CArrays.doubles(m.pointee.body_mass, count: nbody),
            jointNames: names(of: .joint, count: njnt, model: m),
            jointType: CArrays.ints(m.pointee.jnt_type, count: njnt),
            jointBodyID: CArrays.ints(m.pointee.jnt_bodyid, count: njnt),
            jointQposAddress: CArrays.ints(m.pointee.jnt_qposadr, count: njnt),
            jointDofAddress: CArrays.ints(m.pointee.jnt_dofadr, count: njnt),
            jointLimited: CArrays.bools(m.pointee.jnt_limited, count: njnt),
            jointRangeLow: CArrays.column(m.pointee.jnt_range, rows: njnt, width: 2, index: 0),
            jointRangeHigh: CArrays.column(m.pointee.jnt_range, rows: njnt, width: 2, index: 1),
            qpos0: CArrays.doubles(m.pointee.qpos0, count: nq),
            actuatorNames: names(of: .actuator, count: nu, model: m),
            actuatorTransmissionType: CArrays.ints(m.pointee.actuator_trntype, count: nu),
            actuatorTransmissionID: CArrays.intColumn(m.pointee.actuator_trnid, rows: nu, width: 2, index: 0),
            actuatorControlLimited: CArrays.bools(m.pointee.actuator_ctrllimited, count: nu),
            actuatorControlRangeLow: CArrays.column(m.pointee.actuator_ctrlrange, rows: nu, width: 2, index: 0),
            actuatorControlRangeHigh: CArrays.column(m.pointee.actuator_ctrlrange, rows: nu, width: 2, index: 1),
            actuatorForceLimited: CArrays.bools(m.pointee.actuator_forcelimited, count: nu),
            actuatorForceRangeLow: CArrays.column(m.pointee.actuator_forcerange, rows: nu, width: 2, index: 0),
            actuatorForceRangeHigh: CArrays.column(m.pointee.actuator_forcerange, rows: nu, width: 2, index: 1),
            actuatorGainType: CArrays.ints(m.pointee.actuator_gaintype, count: nu),
            actuatorBiasType: CArrays.ints(m.pointee.actuator_biastype, count: nu),
            actuatorGainParameter0: CArrays.column(
                m.pointee.actuator_gainprm, rows: nu, width: MuJoCoPhysics.gainParametersPerActuator, index: 0),
            actuatorGear0: CArrays.column(m.pointee.actuator_gear, rows: nu, width: 6, index: 0),
            sensorNames: names(of: .sensor, count: nsensor, model: m),
            sensorAddress: CArrays.ints(m.pointee.sensor_adr, count: nsensor),
            sensorDimension: CArrays.ints(m.pointee.sensor_dim, count: nsensor),
            cameraNames: names(of: .camera, count: ncam, model: m),
            geomBodyID: CArrays.ints(m.pointee.geom_bodyid, count: ngeom),
            bodyRootID: CArrays.ints(m.pointee.body_rootid, count: nbody))
    }

    // MARK: ActuatorParameterWriting

    /// Writes nu numbers into actuator_gainprm column 0 (env.py:2843-2844).
    public func setActuatorGainParameter0(_ values: [Double]) throws {
        let (m, _) = try loadedPointers()
        let nu = Int(m.pointee.nu)
        try CArrays.check(values, name: "actuatorGainParameter0", count: nu)
        guard let gain = m.pointee.actuator_gainprm else { throw PhysicsError.notLoaded }
        for actuator in 0 ..< nu {
            gain[actuator * MuJoCoPhysics.gainParametersPerActuator] = values[actuator]
        }
    }

    /// Writes nu numbers each into actuator_forcerange columns 0 and 1 (env.py:2845). The control
    /// clip bounds stay those read at load, as MIMo's action space does.
    public func setActuatorForceRange(low: [Double], high: [Double]) throws {
        let (m, _) = try loadedPointers()
        let nu = Int(m.pointee.nu)
        try CArrays.check(low, name: "actuatorForceRangeLow", count: nu)
        try CArrays.check(high, name: "actuatorForceRangeHigh", count: nu)
        guard let range = m.pointee.actuator_forcerange else { throw PhysicsError.notLoaded }
        for actuator in 0 ..< nu {
            range[actuator * 2] = low[actuator]
            range[actuator * 2 + 1] = high[actuator]
        }
    }

    // MARK: ScratchKinematicsQuerying

    /// The world position (metres) of `body` with each joint id in `jointAngles` at its angle and
    /// every other qpos entry at qpos0: mj_resetData on the scratch mjData, the angles written at
    /// jnt_qposadr, mj_forward, xpos[body] read (action_decoder.py:224-238). The state mjData is
    /// not touched.
    public func scratchBodyPosition(ofBody body: Int, withJointAngles jointAngles: [Int: Double]) throws -> [Double] {
        let (m, _) = try loadedPointers()
        let nbody = Int(m.pointee.nbody)
        let njnt = Int(m.pointee.njnt)
        guard (0 ..< nbody).contains(body) else {
            throw PhysicsError.idOutOfRange(kind: .body, id: body, count: nbody)
        }
        let scratch = try scratchBuffer(model: m)
        mj_resetData(m, scratch)
        guard let qposAddress = m.pointee.jnt_qposadr, let scratchQpos = scratch.pointee.qpos else {
            throw PhysicsError.notLoaded
        }
        for (joint, angle) in jointAngles.sorted(by: { $0.key < $1.key }) {
            guard (0 ..< njnt).contains(joint) else {
                throw PhysicsError.idOutOfRange(kind: .joint, id: joint, count: njnt)
            }
            guard angle.isFinite else {
                throw PhysicsError.nonFinite(name: "jointAngles", index: joint)
            }
            scratchQpos[Int(qposAddress[joint])] = angle
        }
        mj_forward(m, scratch)
        return CArrays.doubles(scratch.pointee.xpos, offset: body * 3, count: 3)
    }

    // MARK: Private

    /// mjtObj numbers (Python mujoco 3.3.0: mjOBJ_BODY 1, JOINT 3, GEOM 5, CAMERA 7, ACTUATOR 19,
    /// SENSOR 20).
    static func objectType(_ kind: MuJoCoObjectKind) -> Int32 {
        switch kind {
        case .body: return 1
        case .joint: return 3
        case .geom: return 5
        case .camera: return 7
        case .actuator: return 19
        case .sensor: return 20
        }
    }

    private func loadedPointers() throws -> (UnsafeMutablePointer<mjModel>, UnsafeMutablePointer<mjData>) {
        guard let m = model, let d = data else { throw PhysicsError.notLoaded }
        return (m, d)
    }

    /// A flat copy of one mjtNum array of the state, its count computed from the model sizes.
    private func dataArray(
        count: (PhysicsSizes) -> Int, pick: (UnsafeMutablePointer<mjData>) -> UnsafeMutablePointer<Double>?
    ) -> [Double] {
        guard model != nil, let d = data else { return [] }
        return CArrays.doubles(pick(d), count: count(sizes))
    }

    private func names(of kind: MuJoCoObjectKind, count: Int, model m: UnsafeMutablePointer<mjModel>) -> [String] {
        let type = MuJoCoPhysics.objectType(kind)
        return (0 ..< count).map { index in
            guard let cName = mj_id2name(m, type, Int32(index)) else { return "" }
            return String(cString: cName)
        }
    }

    private func sceneBuffer(model m: UnsafeMutablePointer<mjModel>) throws -> UnsafeMutablePointer<mjvScene> {
        if let existing = abstractScene {
            return existing
        }
        let scene = UnsafeMutablePointer<mjvScene>.allocate(capacity: 1)
        scene.initialize(to: mjvScene())
        mjv_defaultScene(scene)
        mjv_makeScene(m, scene, Int32(maxSceneGeoms))
        guard scene.pointee.maxgeom > 0 else {
            mjv_freeScene(scene)
            scene.deinitialize(count: 1)
            scene.deallocate()
            throw PhysicsError.sceneAllocationFailed(requested: maxSceneGeoms)
        }
        abstractScene = scene
        return scene
    }

    private func scratchBuffer(model m: UnsafeMutablePointer<mjModel>) throws -> UnsafeMutablePointer<mjData> {
        if let existing = scratchData {
            return existing
        }
        guard let scratch = mj_makeData(m) else { throw PhysicsError.dataAllocationFailed }
        scratchData = scratch
        return scratch
    }

    /// Frees the abstract scene, both mjData and the mjModel, and clears the load's values.
    private func release() {
        if let scene = abstractScene {
            mjv_freeScene(scene)
            scene.deinitialize(count: 1)
            scene.deallocate()
        }
        abstractScene = nil
        if let scratch = scratchData {
            mj_deleteData(scratch)
        }
        scratchData = nil
        if let d = data {
            mj_deleteData(d)
        }
        data = nil
        if let m = model {
            mj_deleteModel(m)
        }
        model = nil
        sceneURL = nil
        controlLowerBounds = []
        controlUpperBounds = []
        physicsStepCount = 0
    }
}

/// Copies between MuJoCo's C arrays and Swift arrays. One responsibility: the pointer
/// arithmetic, with a nil pointer or a count of 0 giving an empty array.
enum CArrays {
    /// `count` doubles starting `offset` numbers after `pointer`.
    static func doubles(_ pointer: UnsafeMutablePointer<Double>?, offset: Int = 0, count: Int) -> [Double] {
        guard let pointer = pointer, count > 0 else { return [] }
        return Array(UnsafeMutableBufferPointer(start: pointer + offset, count: count))
    }

    /// `count` C ints as Swift Ints.
    static func ints(_ pointer: UnsafeMutablePointer<Int32>?, count: Int) -> [Int] {
        guard let pointer = pointer, count > 0 else { return [] }
        return UnsafeMutableBufferPointer(start: pointer, count: count).map { Int($0) }
    }

    /// `count` mjtByte flags as Bools (nonzero is true).
    static func bools(_ pointer: UnsafeMutablePointer<UInt8>?, count: Int) -> [Bool] {
        guard let pointer = pointer, count > 0 else { return [] }
        return UnsafeMutableBufferPointer(start: pointer, count: count).map { $0 != 0 }
    }

    /// Column `index` of a row-major rows x width mjtNum table.
    static func column(_ pointer: UnsafeMutablePointer<Double>?, rows: Int, width: Int, index: Int) -> [Double] {
        guard let pointer = pointer, rows > 0, width > 0, (0 ..< width).contains(index) else { return [] }
        return (0 ..< rows).map { pointer[$0 * width + index] }
    }

    /// Column `index` of a row-major rows x width int table.
    static func intColumn(_ pointer: UnsafeMutablePointer<Int32>?, rows: Int, width: Int, index: Int) -> [Int] {
        guard let pointer = pointer, rows > 0, width > 0, (0 ..< width).contains(index) else { return [] }
        return (0 ..< rows).map { Int(pointer[$0 * width + index]) }
    }

    /// Copies `values` into the C array at `pointer` (which holds at least values.count numbers).
    static func write(_ values: [Double], into pointer: UnsafeMutablePointer<Double>?) {
        guard let pointer = pointer else { return }
        for (index, value) in values.enumerated() {
            pointer[index] = value
        }
    }

    /// Throws PhysicsError.wrongCount when `values` does not hold `count` numbers and
    /// PhysicsError.nonFinite at the first NaN or infinity.
    static func check(_ values: [Double], name: String, count: Int) throws {
        guard values.count == count else {
            throw PhysicsError.wrongCount(name: name, expected: count, found: values.count)
        }
        if let bad = values.firstIndex(where: { !$0.isFinite }) {
            throw PhysicsError.nonFinite(name: name, index: bad)
        }
    }

    /// The text of a NUL-terminated C character buffer, read as UTF-8 up to the first NUL.
    static func text(_ buffer: [CChar]) -> String {
        let bytes = buffer.prefix { $0 != 0 }.map { UInt8(truncatingIfNeeded: $0) }
        return String(decoding: bytes, as: UTF8.self)
    }
}
