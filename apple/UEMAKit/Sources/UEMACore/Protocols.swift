// Protocols.swift: the nine protocols every later UEMAKit module implements or takes as an
// init parameter (Physics, SceneLoading, SenseReader, Tokenizer, WeightsLoading,
// ForwardModeling, Planning, CommandDecoding, EyeRendering), the three reading protocols
// that split them (PhysicsReading, SceneGeomReading, and the Physics step defaults), and
// the value types that cross them. Work item WI-20, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; this is the work item's main
// file. Together with SenseName.swift, Command.swift and ModelStep.swift it is the module
// UEMACore, which depends on nothing but Foundation (Package.swift, WI-17).
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). No MLX Swift, SceneKit, Metal or MuJoCo symbol is
// used here: the module names only plain Swift and Foundation types, so that a fake of any
// protocol compiles in a test target without those libraries (plan rows WI-27, WI-32).
//
// WHO IMPLEMENTS WHAT (plan rows; specification section 5 names in parentheses):
//   Physics, PhysicsReading, SceneGeomReading  <- UEMAPhysics.MuJoCoPhysics (WI-25;
//       PhysicsEngine, PhysicsReader)
//   SceneLoading                                <- UEMAScene.SceneBundle (WI-26; SceneBundle)
//   SenseReader                                 <- UEMASenses: ProprioceptionReader,
//       VestibularReader, MotorReader (WI-27), the tactile reader (WI-28), the vision reader
//       over an EyeRendering (WI-29), audio (WI-16's audio_render.md) (SenseSource)
//   EyeRendering                                <- UEMASenses.SceneKitEyeRenderer (WI-29;
//       SceneRenderer)
//   Tokenizer                                   <- UEMATokenizer, one per sense (WI-30)
//   WeightsLoading                              <- UEMAModel.SafetensorsLoader (WI-31;
//       WeightsLoader)
//   ForwardModeling                             <- UEMAModel: encoders, transformer, heads,
//       motor controller, wellness (WI-35, WI-36; SenseEncoder, ForwardModel,
//       PredictionHeads, MotorController, WellnessScorer)
//   Planning                                    <- UEMAPlanner.Planner with GoalRing and
//       HandDescent (WI-39; GoalRing, Planner, HandDescent)
//   CommandDecoding                             <- UEMACommand.CommandDecoder (WI-37;
//       CommandDecoder)
//   The session loop (apple/UEMAMind RunViewModel, WI-32) takes one of each as an init
//   parameter and calls them in the sequence of plan section 2.
//
// FIRST BUILD ON A MAC (this work item's acceptance: "UEMACoreTests pass (234 numbers per
// Command; six active senses; 192 per Token)"):
//  1. Run the plan's section 8 steps before this one first: `bash
//     apple/scripts/build_mujoco_xcframework.sh` exit 0 (WI-18), then `cd apple/UEMAKit &&
//     swift package resolve` exit 0 (WI-17). The package manifest does not load without the
//     MuJoCo binary target, so UEMACore cannot be built alone before them even though it
//     depends on nothing but Foundation.
//  2. `cd apple/UEMAKit && xcodebuild -scheme UEMAKit-Package build -destination
//     'platform=macOS'` : exit 0. A compile error in this module is corrected here; a
//     compile error in a later module that names a requirement of this file means that
//     module's conformance or fake is out of date with this contract, and this file wins.
//  3. `xcodebuild -scheme UEMAKit-Package test -destination 'platform=macOS'
//     -only-testing:UEMACoreTests` : exit 0 with every test passing. The tests are
//     UEMACorePackageTests (1, WI-17), CommandTests (8), SenseNameTests (4), ModelStepTests
//     (4) and ProtocolsTests (10): 27 in all.
//  4. Constructs to check first if step 2 fails, each used on purpose: the primary
//     associated type `protocol WeightsLoading<Weights>` (Swift 5.7 or newer); the
//     parenthesised range literals `(-0.785) ... (0.785)`; `Sendable` on structs holding a
//     Foundation `URL`; a custom `init(from:)` that delegates to a throwing initializer;
//     protocol extension defaults that take a non-escaping closure and pass it on.
//     Tests/UEMACoreTests/ProtocolsTests.swift holds one fake of every protocol here: a
//     compile error in a fake names the requirement to check first.
//  4a. `EyeImage` has one declaration, in this file: Sources/UEMASenses/Vision/CameraSpec.swift
//     (WI-29) adds `extension EyeImage` and its own `EyeImageDifference` (both read November 4,
//     2026). If a later module declares another public type named `EyeImage`, a module that
//     imports both and writes the bare name gets an "ambiguous for type lookup" error; the
//     correction is in the later module, never a rename here.
//  5. Record each step's command, exit code and clock time (AM/PM Pacific) in
//     apple/FIRST_BUILD_RECORD.md (WI-41).
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave:
// - docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html row WI-20: "protocols Physics,
//   SceneLoading, SenseReader, Tokenizer, WeightsLoading, ForwardModeling, Planning,
//   CommandDecoding, EyeRendering. Every later module takes these as init parameters."; row
//   WI-25: "MuJoCoPhysics implements Physics: load(bundle) by mj_loadXML;
//   stepModelStep(controls:) runs 5 env steps of 2 x (mj_step + mj_rnePostConstraint) at
//   the scene's 0.005 s timestep; readers for the arrays of WI-19 and the contacts (ncon,
//   mj_contactForce); writes ctrl clipped to -1..1."; row WI-27: "proprioception rows from
//   xquat, xpos, actuator_force; vestibular 29 numbers from sensordata, qpos, qvel,
//   actuator_force"; row WI-28: "Per physics substep: for each of data.ncon contacts,
//   mj_contactForce rotated into the body frame"; row WI-29: "GeomMirror maps each geom
//   (type, size, position, rotation, rgba) to an SCNNode ... renders offscreen for the five
//   cameras (eye_left and eye_right 60 degrees, 256 px; fovea_left and fovea_right 24
//   degrees, 288 px; ring_left 150 degrees, 256 px) into RGB uint8 images"; row WI-31:
//   "every key and shape is checked against weight_manifest.json ... a missing or extra key
//   is an error naming it"; row WI-32: "per model step physics 10 steps, read, tokenize,
//   model, command (planner every 10th), decode, write"; row WI-39: "Planner holds one
//   command (234 numbers) and runs Adam at learning rate 0.1 for 20 updates with MLX's
//   gradient, scoring the start and every update; ... best-scored evaluation selected per
//   body".
// - docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html section 5 (the protocol table quoted
//   above), REQ-B-3: "a tactile read every physics substep [I2 env.py:2358-2360]", REQ-B-12
//   (planner inputs and outputs), REQ-B-13: "Input: the command (26 x 9), the handle mask
//   [26] and the 44 finger joint targets ... Output: ctrl[92]" (the released scene has nu = 90, `PhysicsSizes.released`; the
//   specification's 92 is corrected here, not followed), REQ-B-14 (the sense panel's
//   "prediction error (mean squared difference)"), REQ-B-17: "the only gradient is the
//   planner's with respect to the command".
// - https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html (mjContact):
//   "mjtNum dist; // distance between nearest points; neg: penetration"; "mjtNum pos[3]; //
//   position of contact point: midpoint between geoms"; "mjtNum frame[9]; // normal is in
//   [0-2], points from geom[0] to geom[1]"; "int geom[2]; // geom ids; -1 for flex".
//   (mjvGeom): "int type; // geom type (mjtGeom)"; "float size[3]; // size parameters";
//   "float pos[3]; // Cartesian position"; "float mat[9]; // Cartesian orientation";
//   "float rgba[4]; // color and transparency".
// - https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html: mj_contactForce
//   "void mj_contactForce(const mjModel* m, const mjData* d, int id, mjtNum result[6]);"
//   "Extract 6D force:torque given contact id, in the contact frame."; mj_step "Advance
//   simulation, use control callback to obtain external force and control.";
//   mj_rnePostConstraint "Recursive Newton Euler with final computed forces and
//   accelerations."; mj_objectVelocity "Compute object 6D velocity (rot:lin) in
//   object-centered frame, world/local orientation."
// - apple/UEMAKit/Specs/proprio_rows.md section 2: velocity rows 7-9 are
//   "`mj_objectVelocity(model, data, mjOBJ_BODY, b, vel6, 0)[3:6] * 100` cm/s, world frame,
//   at the body origin (:210-211; flg_local 0)"; section 3: the fatigue model is "advanced
//   once per env step before the rows are read".
// - apple/UEMAKit/Specs/vestibular_fields.md section 1: "computed every env step"; "The
//   model reads ONE vector per model step, the last env step's".
// - apple/UEMAKit/Specs/planner_loss.md section 1 (planning condition, goal ring), section 3
//   ("21 evaluations, 20 updates"; "a non-finite objective on the start raises, later it
//   stops the search"), section 4 (the hand descent's 44 joint goals), section 6 point D:
//   "The 21 scores recorded per event are `parts[:, 0]` per iteration".
// - apple/UEMAKit/Specs/command_decoder.md section 5 point C: "The receipt's `motor_contract`
//   reported by the world is `MOTOR_CONTRACT + '+' + FINGER_SERVO_CONTRACT`"; section 2:
//   MOTOR_CONTRACT 'ancestor_arm_axes_signed_limits_v3', FINGER_SERVO_CONTRACT
//   'finger_joint_servo_v1'.
// - apple/UEMAKit/Sources/UEMAModel/Resources/weight_manifest.json (WI-12): a tensor entry
//   is {"key": "contact_head.ctx2.bias", "shape": [256], "dtype": "F32", "elements": 256,
//   "bytes": 1024, "component": "contact_head", "kind": "parameter", "inference_set": false};
//   counts.tensors 802, counts.parameter_elements 18131745.
// - apple/UEMAKit/Tests/UEMASensesTests/Fixtures/eyes/step_000492800/cameras.json (WI-14):
//   cameras eye_left (id 1, body left_eye, 256 px, fovy 60.0), eye_right (id 4, right_eye,
//   256, 60.0), fovea_left (2, left_eye, 288, 24.0), fovea_right (5, right_eye, 288, 24.0),
//   ring_left (3, left_eye, 256, 150.0); each with "pos_m" and "xmat_row_major"; image_format
//   "PNG, RGB 8-bit, rows top-down, columns left-to-right".
// - Source/Python/mimo_bridge/env.py lines 3140-3149: "The trainer now says which frames it
//   will read (want_render), so every render is used and every model step gets a fresh
//   image."; "do_render = (force_render or self.frame % self.render_every == 0) if
//   want_render is None else bool(want_render or force_render)". The plan's rows WI-29 and
//   section 2 say "every 2nd model step"; the two readings differ, so this file fixes no
//   render cadence: the session (WI-32) owns it and the WI-29 builder reads train_mimo's
//   want_render rule.
// - apple/UEMAKit/Specs/proprio_rows.md section 5 step 3: the motor encoder's input is
//   "[achieved 10, command 9]" where "the LIVE frame's command columns are zeros" and "the
//   PREVIOUS model step's motor slot in the context buffer is rewritten with
//   [last_achieved[..., :10], command_for_encoder(exec_t)]".

import Foundation

// MARK: - Arrays that cross the sense, tokenizer and model stages

/// A float32 array with its shape, row-major (the last index moves fastest, as numpy and
/// MLX store it). `values.count` equals the product of `shape`; a scalar has the shape [].
public struct ShapedArray: Equatable, Hashable, Codable, Sendable {
    /// The extent of each dimension, outermost first.
    public let shape: [Int]

    /// The numbers, row-major.
    public let values: [Float]

    /// Wraps `values` with `shape`; a negative extent or a product that is not
    /// `values.count` throws `UEMACoreError.invalidShape`.
    public init(shape: [Int], values: [Float]) throws {
        guard ShapedArray.elementCount(of: shape) == values.count else {
            throw UEMACoreError.invalidShape(type: "ShapedArray", shape: shape, valueCount: values.count)
        }
        self.shape = shape
        self.values = values
    }

    /// An all-zero array of `shape`; every extent must be 0 or more.
    public init(zerosOfShape shape: [Int]) {
        precondition(shape.allSatisfy { $0 >= 0 }, "a shape has no negative extent; got \(shape)")
        self.shape = shape
        self.values = Array(repeating: 0, count: shape.reduce(1, *))
    }

    /// The product of `shape`; nil when an extent is negative or the product overflows `Int`
    /// (a corrupt shape read from a file must throw, not stop the process).
    private static func elementCount(of shape: [Int]) -> Int? {
        var product = 1
        for extent in shape {
            guard extent >= 0 else { return nil }
            let (next, overflow) = product.multipliedReportingOverflow(by: extent)
            guard !overflow else { return nil }
            product = next
        }
        return product
    }

    /// The number of values: the product of the shape.
    public var count: Int { values.count }

    /// The number of dimensions.
    public var rank: Int { shape.count }

    private enum CodingKeys: String, CodingKey {
        case shape
        case values
    }

    /// Decodes through the shape check.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            shape: container.decode([Int].self, forKey: .shape),
            values: container.decode([Float].self, forKey: .values))
    }
}

/// One sense's raw arrays for one model step as its `SenseReader` produced them (plan
/// section 2, stage 2), before the tokenizer's divisors. `primary` is the sense's main
/// array (proprioception [26, 12]; motor [26, 10] achieved; vestibular [29]; tactile
/// [n, 10] patch fields, n the patch count (53 shipped); vision [890,880], the RGB bytes of
/// eye_left, eye_right, fovea_left and fovea_right end to end (`VisionReadingLayout`,
/// VisionReading.swift); audio [256, 2], each ear's linear magnitude per frequency bin); `auxiliary`
/// is the second array of a sense that has one (tactile [n, 10]; motor [24]; vision [2, 2], each
/// eye's yaw and pitch in radians; audio [6], the render's raw metadata left_rms, right_rms,
/// ild_db, itd_ms, azimuth in radians, front_back, of which UEMATokenizer's AudioTokenizer passes
/// [ild_db / 20, itd_ms] to the encoder: UEMASenses AudioReader.swift); nil otherwise. `isFresh` is false when
/// the reader had no new reading this model step (vision on a model step without a render;
/// vestibular with its `valid` flag 0), which the Python trainer marks "stale".
public struct SenseReading: Equatable, Codable, Sendable {
    /// The sense this reading belongs to.
    public let sense: SenseName

    /// The main array.
    public let primary: ShapedArray

    /// The second array of a sense that has one; nil otherwise.
    public let auxiliary: ShapedArray?

    /// False for a reading the reader could not refresh this model step.
    public let isFresh: Bool

    /// A reading of `sense`.
    public init(sense: SenseName, primary: ShapedArray, auxiliary: ShapedArray? = nil, isFresh: Bool = true) {
        self.sense = sense
        self.primary = primary
        self.auxiliary = auxiliary
        self.isFresh = isFresh
    }
}

/// One sense's encoder input for one model step as its `Tokenizer` produced it (plan
/// section 2, stage 3): proprioception [312]; motor [26, 19]; vestibular [28]; tactile
/// [32, 19]; vision patches [N, 768] with `auxiliary` angles [N, 4]; audio [256, 2] with
/// `auxiliary` [2] (specification REQ-B-4 to REQ-B-7, REQ-B-9). The two auxiliary widths match
/// the input layers in weight_manifest.json: vision.angular_pos_enc.mlp.0.weight [192, 4] and
/// audio.spatial_proj.weight [192, 2]; the audio [2] is the first 2 of the Python
/// tokenizer's 5 metadata numbers (mimo_bridge/lem_model.py:2476-2477). `isFresh` is carried from the
/// reading.
public struct EncoderInput: Equatable, Codable, Sendable {
    /// The sense this input belongs to.
    public let sense: SenseName

    /// The main array.
    public let primary: ShapedArray

    /// The second array of a sense that has one; nil otherwise.
    public let auxiliary: ShapedArray?

    /// False when the reading it came from was not refreshed this model step.
    public let isFresh: Bool

    /// An encoder input of `sense`.
    public init(sense: SenseName, primary: ShapedArray, auxiliary: ShapedArray? = nil, isFresh: Bool = true) {
        self.sense = sense
        self.primary = primary
        self.auxiliary = auxiliary
        self.isFresh = isFresh
    }
}

// MARK: - Physics

/// The sizes of the loaded MuJoCo model. On the released crib scene (`released`, measured with
/// Python mujoco 3.3.0 on Sources/UEMAScene/Resources/crib_age24_trunk_v1/scene.xml, November 4,
/// 2026, and by MuJoCoPhysicsTests on the same file): nq 146, nv 136, nu 90, nbody 76, njnt 104,
/// nsensordata 168 (from nsensor 56 sensors), ngeom 188, ncam 6.
public struct PhysicsSizes: Equatable, Hashable, Codable, Sendable {
    /// Generalized coordinates (mjModel.nq).
    public let nq: Int
    /// Degrees of freedom (mjModel.nv).
    public let nv: Int
    /// Actuators (mjModel.nu).
    public let nu: Int
    /// Bodies, the world body included (mjModel.nbody).
    public let nbody: Int
    /// Joints (mjModel.njnt).
    public let njnt: Int
    /// Sensor values (mjModel.nsensordata).
    public let nsensordata: Int
    /// Geoms (mjModel.ngeom).
    public let ngeom: Int
    /// Cameras (mjModel.ncam).
    public let ncam: Int

    /// The sizes of the released crib scene (this type's doc comment gives the measurement).
    public static let released = PhysicsSizes(
        nq: 146, nv: 136, nu: 90, nbody: 76, njnt: 104, nsensordata: 168, ngeom: 188, ncam: 6)

    /// The sizes as read from the loaded model.
    public init(nq: Int, nv: Int, nu: Int, nbody: Int, njnt: Int, nsensordata: Int, ngeom: Int, ncam: Int) {
        self.nq = nq
        self.nv = nv
        self.nu = nu
        self.nbody = nbody
        self.njnt = njnt
        self.nsensordata = nsensordata
        self.ngeom = ngeom
        self.ncam = ncam
    }
}

/// One contact of the current physics step, copied from MuJoCo's mjContact plus
/// mj_contactForce (MuJoCo API reference, quoted in this file's header): `geomIds` the two
/// geom ids ("-1 for flex"); `distance` "distance between nearest points; neg: penetration";
/// `position` the "position of contact point: midpoint between geoms" (3, metres, world);
/// `frame` the 9 numbers of the contact frame, "normal is in [0-2], points from geom[0] to
/// geom[1]"; `force` the "6D force:torque ... in the contact frame" that mj_contactForce
/// gives (force 3, then torque 3).
public struct Contact: Equatable, Hashable, Codable, Sendable {
    /// The two geom ids (mjContact.geom), -1 for a flex.
    public let geomIds: [Int]
    /// Distance between the nearest points in metres; negative is penetration (mjContact.dist).
    public let distance: Double
    /// The contact point, 3 numbers, metres, world frame (mjContact.pos).
    public let position: [Double]
    /// The contact frame, 9 numbers; the normal is the first 3 (mjContact.frame).
    public let frame: [Double]
    /// Force 3 then torque 3 in the contact frame (mj_contactForce), newtons and newton metres.
    public let force: [Double]

    /// A contact; counts other than geomIds 2, position 3, frame 9, force 6 throw
    /// `UEMACoreError.wrongCount`.
    public init(geomIds: [Int], distance: Double, position: [Double], frame: [Double], force: [Double]) throws {
        guard geomIds.count == 2 else {
            throw UEMACoreError.wrongCount(type: "Contact.geomIds", expected: 2, found: geomIds.count)
        }
        guard position.count == 3 else {
            throw UEMACoreError.wrongCount(type: "Contact.position", expected: 3, found: position.count)
        }
        guard frame.count == 9 else {
            throw UEMACoreError.wrongCount(type: "Contact.frame", expected: 9, found: frame.count)
        }
        guard force.count == 6 else {
            throw UEMACoreError.wrongCount(type: "Contact.force", expected: 6, found: force.count)
        }
        self.geomIds = geomIds
        self.distance = distance
        self.position = position
        self.frame = frame
        self.force = force
    }

    private enum CodingKeys: String, CodingKey {
        case geomIds
        case distance
        case position
        case frame
        case force
    }

    /// Decodes through the count checks.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            geomIds: container.decode([Int].self, forKey: .geomIds),
            distance: container.decode(Double.self, forKey: .distance),
            position: container.decode([Double].self, forKey: .position),
            frame: container.decode([Double].self, forKey: .frame),
            force: container.decode([Double].self, forKey: .force))
    }
}

/// The kinds of named MuJoCo object a `PhysicsReading` resolves to an id.
public enum MuJoCoObjectKind: String, Codable, Hashable, Sendable {
    case body
    case joint
    case geom
    case actuator
    case sensor
    case camera
}

/// Read access to the loaded MuJoCo model and its current state: the arrays the sense
/// readers and the command decoder read (plan rows WI-25, WI-27, WI-28; command_decoder.md
/// section 1, "MuJoCo state"). Every array is a flat copy in MuJoCo's row-major layout and
/// its units (metres, radians, seconds, newtons): `xpos` is nbody x 3, `xquat` nbody x 4 in
/// MuJoCo's w x y z sequence (proprio_adapter.py:207-208 reorders it to x y z w), `xmat`
/// nbody x 9, `xaxis` and `xanchor` njnt x 3, `actuatorForce` and `control` nu,
/// `sensordata` nsensordata. The time-invariant model constants the decoder and the
/// vestibular reader also need (jnt_range, jnt_qposadr, jnt_dofadr, actuator_trnid,
/// actuator_ctrlrange, actuator_gear, body_mass and the rest) are read through
/// UEMAPhysics's own model-description protocol (WI-25), beside its CMuJoCo accessors.
/// Class-bound so that one instance is shared by reference between the physics, the
/// readers and the decoder.
public protocol PhysicsReading: AnyObject {
    /// The sizes of the loaded model.
    var sizes: PhysicsSizes { get }

    /// Simulated time in seconds (mjData.time).
    var time: Double { get }

    /// Generalized positions, nq.
    var qpos: [Double] { get }

    /// Generalized velocities, nv.
    var qvel: [Double] { get }

    /// Sensor values, nsensordata.
    var sensordata: [Double] { get }

    /// Body positions, nbody x 3, metres, world frame.
    var xpos: [Double] { get }

    /// Body orientations, nbody x 4, quaternion w x y z.
    var xquat: [Double] { get }

    /// Body rotation matrices, nbody x 9, row-major.
    var xmat: [Double] { get }

    /// Joint axes in world frame, njnt x 3.
    var xaxis: [Double] { get }

    /// Joint anchors in world frame, njnt x 3, metres.
    var xanchor: [Double] { get }

    /// Actuator forces, nu.
    var actuatorForce: [Double] { get }

    /// The controls written for the current env step, nu (mjData.ctrl).
    var control: [Double] { get }

    /// The contacts of the current physics step (ncon entries).
    var contacts: [Contact] { get }

    /// The id of the named object of `kind`; nil when the model has no such name.
    func id(of kind: MuJoCoObjectKind, named name: String) -> Int?

    /// The 6D velocity of a body at its origin in the world frame, rotational 3 then linear
    /// 3 (mj_objectVelocity with flg_local 0, proprio_adapter.py:210-211), metres per second
    /// and radians per second.
    func bodyVelocity(bodyId: Int) -> [Double]
}

/// The MuJoCo physics the session drives (plan row WI-25; specification REQ-B-3): load the
/// scene, write the 90 controls, run physics steps, and read the state through
/// `PhysicsReading`. The two step methods with closures have default implementations below
/// in terms of `write(controls:)` and `stepPhysicsStep()`; a concrete type may replace them.
public protocol Physics: PhysicsReading {
    /// Loads the scene (mj_loadXML on the bundle's scene.xml) and makes its data.
    func load(sceneXML url: URL) throws

    /// True once `load(sceneXML:)` succeeded.
    var isLoaded: Bool { get }

    /// Resets the state to the model's initial state (mj_resetData) and the time to 0.
    func reset() throws

    /// Sets qpos (nq) and qvel (nv), as the fixtures of WI-14 and WI-34 do.
    func setState(qpos: [Double], qvel: [Double]) throws

    /// Writes the 90 controls to mjData.ctrl, each clipped to -1 to 1 (plan row WI-25).
    func write(controls: MotorControls) throws

    /// One physics step: mj_step followed by mj_rnePostConstraint (mimo_env.py:573-575),
    /// 0.005 s of simulated time. MIMo's actuation model runs its substep update before
    /// each (mimo_env.py:617), which the concrete type reproduces.
    func stepPhysicsStep() throws

    /// One env step (0.010 s): writes `controls`, then runs 2 physics steps, calling
    /// `afterEachPhysicsStep` with this physics after each one (the tactile reader's
    /// per-substep contact read, plan row WI-28; specification REQ-B-3). The Python
    /// trainer decodes a fresh ctrl for every env step (command_decoder.md section 1), so
    /// the session calls this 5 times per model step with a fresh decode.
    func stepEnvStep(controls: MotorControls, afterEachPhysicsStep: (any PhysicsReading) throws -> Void) throws

    /// One model step (0.050 s): 5 env steps holding the same `controls` (the plan row
    /// WI-25 signature; for a decode per env step use `stepEnvStep` five times).
    func stepModelStep(controls: MotorControls, afterEachPhysicsStep: (any PhysicsReading) throws -> Void) throws
}

extension Physics {
    public func stepEnvStep(
        controls: MotorControls, afterEachPhysicsStep: (any PhysicsReading) throws -> Void
    ) throws {
        try write(controls: controls)
        for _ in 0 ..< ModelStep.physicsStepsPerEnvStep {
            try stepPhysicsStep()
            try afterEachPhysicsStep(self)
        }
    }

    public func stepModelStep(
        controls: MotorControls, afterEachPhysicsStep: (any PhysicsReading) throws -> Void
    ) throws {
        for _ in 0 ..< ModelStep.envStepsPerModelStep {
            try stepEnvStep(controls: controls, afterEachPhysicsStep: afterEachPhysicsStep)
        }
    }
}

// MARK: - Scene files

/// The exported scene files (plan rows WI-13, WI-15, WI-26): where scene.xml and the skin
/// layout JSON are, and any other file of the bundle by name. The concrete type finds them
/// in the UEMAScene module bundle and checks their hashes (specification REQ-B-2).
public protocol SceneLoading {
    /// The crib scene's scene.xml (WI-15), the file `Physics.load(sceneXML:)` takes.
    var sceneXMLURL: URL { get }

    /// The skin layout JSON of 4,683 points (WI-13), the file the tactile reader takes.
    var skinLayoutURL: URL { get }

    /// Any file of the bundle by its name inside the bundle folder; throws when absent.
    func url(forFile name: String) throws -> URL
}

// MARK: - Senses

/// One sense's reader over the physics (plan section 2, stage 2): it is told every physics
/// step and every env step, and once per model step it returns the sense's raw arrays.
/// The two observe methods default to doing nothing; a reader that integrates per substep
/// (tactile, plan row WI-28) or per env step (proprioception's fatigue model and the
/// vestibular fields, proprio_rows.md section 3, vestibular_fields.md section 1) implements
/// the one it needs. Class-bound: readers keep state between calls.
public protocol SenseReader: AnyObject {
    /// The sense this reader produces.
    var sense: SenseName { get }

    /// Called after each physics step (10 per model step) with the current state.
    func observePhysicsStep(_ physics: any PhysicsReading) throws

    /// Called after each env step (5 per model step) with the current state.
    func observeEnvStep(_ physics: any PhysicsReading) throws

    /// The sense's raw arrays for the model step that just completed.
    func read(_ physics: any PhysicsReading, at step: ModelStep) throws -> SenseReading

    /// Clears any state kept between model steps.
    func reset()
}

extension SenseReader {
    public func observePhysicsStep(_ physics: any PhysicsReading) throws {}

    public func observeEnvStep(_ physics: any PhysicsReading) throws {}

    public func reset() {}
}

/// One sense's tokenizer (plan section 2, stage 3; plan row WI-30): the fixed divisors and
/// cuts that turn a `SenseReading` into the sense's `EncoderInput`. A tokenizer that keeps
/// frozen statistics (the vestibular normaliser's checkpoint mean and variance,
/// vestibular_fields.md section 3) holds them as stored properties.
public protocol Tokenizer {
    /// The sense this tokenizer serves.
    var sense: SenseName { get }

    /// The encoder input of one reading; a reading of another sense or shape throws.
    func tokenize(_ reading: SenseReading) throws -> EncoderInput
}

// MARK: - Weights

/// One row of weight_manifest.json (WI-12): one tensor of model.safetensors by key, shape,
/// dtype ("F32", "F64", "BOOL", "I64"), element count, byte count, component, kind
/// ("parameter" or "buffer") and whether it belongs to the inference set.
public struct WeightsManifestEntry: Equatable, Hashable, Codable, Sendable {
    /// The tensor's key in model.safetensors, such as "contact_head.ctx2.bias".
    public let key: String
    /// The tensor's shape, outermost first.
    public let shape: [Int]
    /// The stored element type: "F32", "F64", "BOOL" or "I64".
    public let dtype: String
    /// The product of `shape`.
    public let elements: Int
    /// The stored size in bytes.
    public let bytes: Int
    /// The model component the tensor belongs to, such as "contact_head".
    public let component: String
    /// "parameter" or "buffer".
    public let kind: String
    /// True when the inference-only session uses the tensor (JSON key "inference_set").
    public let inferenceSet: Bool

    /// A manifest row.
    public init(key: String, shape: [Int], dtype: String, elements: Int, bytes: Int, component: String, kind: String, inferenceSet: Bool) {
        self.key = key
        self.shape = shape
        self.dtype = dtype
        self.elements = elements
        self.bytes = bytes
        self.component = component
        self.kind = kind
        self.inferenceSet = inferenceSet
    }

    private enum CodingKeys: String, CodingKey {
        case key
        case shape
        case dtype
        case elements
        case bytes
        case component
        case kind
        case inferenceSet = "inference_set"
    }
}

/// What a weights load did (plan row WI-31; specification REQ-B-11's pass line "Every
/// manifest tensor is reported loaded or skipped by name"): the file, the keys loaded into
/// the module tree, the keys skipped by name (training-only tensors), and the elements
/// loaded. A missing or extra key is an error thrown by the loader, never a report entry.
public struct WeightsReport: Equatable, Codable, Sendable {
    /// The safetensors file that was read.
    public let fileURL: URL
    /// The keys loaded into the module tree.
    public let loadedKeys: [String]
    /// The keys skipped by name (training-only tensors).
    public let skippedKeys: [String]
    /// The sum of the loaded tensors' element counts.
    public let loadedElements: Int

    /// A load report.
    public init(fileURL: URL, loadedKeys: [String], skippedKeys: [String], loadedElements: Int) {
        self.fileURL = fileURL
        self.loadedKeys = loadedKeys
        self.skippedKeys = skippedKeys
        self.loadedElements = loadedElements
    }
}

/// Loads model.safetensors (73,119,988 bytes; not in git) into the arrays the model takes,
/// checked against the manifest (plan row WI-31). `Weights` is the concrete arrays type
/// (a dictionary of MLX arrays in UEMAModel); UEMACore does not name MLX, so the type is an
/// associated type of the conformer.
public protocol WeightsLoading<Weights> {
    /// The loaded arrays' type.
    associatedtype Weights

    /// Reads the file at `url`, checks every key and shape against the manifest, and returns
    /// the arrays with the report; a missing or extra key throws an error naming it.
    func load(from url: URL) throws -> (weights: Weights, report: WeightsReport)
}

// MARK: - The model

/// What one model step of the model produced (plan section 2, stages 4 to 7): the six
/// encoder tokens of this model step, the six predicted tokens of the next model step, the
/// motor controller's command (scale `.normalized`), and the wellness score (the wellness
/// channel sum) of this model step for the goal ring.
public struct ModelOutput: Equatable, Codable, Sendable {
    /// The model step these outputs belong to.
    public let step: ModelStep

    /// The encoder tokens of this model step, one per active sense.
    public let tokens: TokenSet

    /// The prediction heads' tokens for the next model step, one per active sense.
    public let predictedTokens: TokenSet

    /// The motor controller's command, positions in position_range units.
    public let command: Command

    /// The wellness channel sum of this model step.
    public let wellness: Float

    public init(step: ModelStep, tokens: TokenSet, predictedTokens: TokenSet, command: Command, wellness: Float) {
        self.step = step
        self.tokens = tokens
        self.predictedTokens = predictedTokens
        self.command = command
        self.wellness = wellness
    }
}

/// The released model run forward for one model step (plan rows WI-35, WI-36): the six
/// encoders, the transformer over the 768-token context, the prediction heads, the motor
/// controller and the wellness sum, with frozen weights (specification REQ-B-17). The
/// planner's gradient with respect to the command goes through the concrete MLX model in
/// UEMAModel and UEMAPlanner, not through this protocol. Class-bound: the context buffer is
/// state.
public protocol ForwardModeling: AnyObject {
    /// The context in model steps: 128.
    var contextLength: Int { get }

    /// The model steps the context holds so far, 0 to `contextLength`.
    var filledContextSteps: Int { get }

    /// Encodes `inputs` (one per active sense), rewrites the previous model step's motor slot
    /// with `previousExecutedCommand` (scale `.centimetres`; nil on the first model step;
    /// proprio_rows.md section 5 step 3), runs the transformer, the heads, the motor
    /// controller and wellness, and advances the context by one model step.
    func step(inputs: [SenseName: EncoderInput], previousExecutedCommand: Command?, at step: ModelStep) throws -> ModelOutput

    /// Empties the context.
    func reset()
}

// MARK: - The planner

/// What one planning event produced (specification REQ-B-12; plan row WI-39): the selected
/// raw command (scale `.normalized`, before the execution transform), the score of the
/// start and of each update (21 when the search ran to the end; fewer when a non-finite
/// objective stopped it, planner_loss.md section 3 step 8), the index of the selected
/// evaluation, the goal ring entry used as the goal, and the hand descent's 44 finger
/// joint goals when it ran.
public struct PlanResult: Equatable, Codable, Sendable {
    /// Evaluations per full event: 21 (the start and 20 updates).
    public static let evaluationsPerEvent = 21

    /// Adam updates per event: 20 (`n_steps` 20, mimo_bridge/lem_model.py:1211).
    public static let updatesPerEvent = 20

    /// Adam's step size: 0.1 (`lr` 0.1, mimo_bridge/lem_model.py:1211).
    public static let learningRate: Float = 0.1

    /// The weight of the sum of squared strengths in the score: 0.01 (`strength_reg` 0.01,
    /// mimo_bridge/lem_model.py:1211; planner_loss.md section 3 step 6).
    public static let strengthRegularisation: Float = 0.01

    /// The selected raw command, positions in position_range units.
    public let command: Command

    /// The total score per evaluation, start first (`parts[:, 0]` per iteration).
    public let scores: [Float]

    /// The index into `scores` of the selected evaluation.
    public let selectedIteration: Int

    /// The model step of the goal ring entry used as the goal.
    public let goalStep: ModelStep

    /// The wellness of that entry.
    public let goalWellness: Float

    /// The hand descent's 44 finger joint goals; nil when it did not run.
    public let fingerGoals: FingerJointGoals?

    /// A plan result; empty scores, a selected index outside them, or a command not at
    /// `.normalized` scale throw.
    public init(command: Command, scores: [Float], selectedIteration: Int, goalStep: ModelStep, goalWellness: Float, fingerGoals: FingerJointGoals?) throws {
        guard command.scale == .normalized else {
            throw UEMACoreError.wrongScale(expected: .normalized, found: command.scale)
        }
        guard !scores.isEmpty, scores.count <= PlanResult.evaluationsPerEvent else {
            throw UEMACoreError.wrongCount(
                type: "PlanResult.scores", expected: PlanResult.evaluationsPerEvent, found: scores.count)
        }
        guard scores.indices.contains(selectedIteration) else {
            throw UEMACoreError.indexOutOfRange(
                type: "PlanResult.selectedIteration", index: selectedIteration, count: scores.count)
        }
        self.command = command
        self.scores = scores
        self.selectedIteration = selectedIteration
        self.goalStep = goalStep
        self.goalWellness = goalWellness
        self.fingerGoals = fingerGoals
    }

    /// The score of the selected evaluation.
    public var selectedScore: Float { scores[selectedIteration] }

    private enum CodingKeys: String, CodingKey {
        case command
        case scores
        case selectedIteration
        case goalStep
        case goalWellness
        case fingerGoals
    }

    /// Decodes through the checks.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            command: container.decode(Command.self, forKey: .command),
            scores: container.decode([Float].self, forKey: .scores),
            selectedIteration: container.decode(Int.self, forKey: .selectedIteration),
            goalStep: container.decode(ModelStep.self, forKey: .goalStep),
            goalWellness: container.decode(Float.self, forKey: .goalWellness),
            fingerGoals: container.decodeIfPresent(FingerJointGoals.self, forKey: .fingerGoals))
    }
}

/// The planner with its goal ring (plan row WI-39; planner_loss.md sections 1 to 4): every
/// model step it records the tokens and wellness into the ring of the last 200 model
/// steps; on a model step whose count is a multiple of 10, once the ring holds 20 entries,
/// `plan(from:at:)` refines a command through the frozen forward model and returns the
/// result. `canPlan(at:)` defaults to that condition. Class-bound: the ring is state.
public protocol Planning: AnyObject {
    /// The entries the goal ring holds now, 0 to 200.
    var goalRingCount: Int { get }

    /// Appends this model step's tokens and wellness to the goal ring
    /// (`self.goal_ring.append`, mimo_bridge/lem_model.py:4271).
    func record(tokens: TokenSet, wellness: Float, at step: ModelStep)

    /// True when a planning event runs at `step`: the ring holds at least 20 entries and
    /// `step.count % 10 == 0` (lem_model.py:4283).
    func canPlan(at step: ModelStep) -> Bool

    /// One planning event from `command` (the motor controller's output of this model step,
    /// scale `.normalized`).
    func plan(from command: Command, at step: ModelStep) throws -> PlanResult

    /// Empties the goal ring.
    func reset()
}

extension Planning {
    public func canPlan(at step: ModelStep) -> Bool {
        goalRingCount >= ModelStep.goalRingMinimumForPlanning && step.isPlanningCandidate
    }
}

// MARK: - The command decoder

/// The command decoder (plan row WI-37; specification REQ-B-13; command_decoder.md): one
/// command at `.centimetres` scale, the handle mask, the gaze pair and the finger joint
/// goals into the 90 motor controls of the current env step, reading the body's state from
/// the physics it was given at init. Class-bound: it keeps its scratch data and the last
/// controls.
public protocol CommandDecoding: AnyObject {
    /// The contract string the Python world reports: MOTOR_CONTRACT + "+" +
    /// FINGER_SERVO_CONTRACT = "ancestor_arm_axes_signed_limits_v3+finger_joint_servo_v1"
    /// (command_decoder.md section 2 and section 5 point C).
    var motorContract: String { get }

    /// The 90 controls for this env step. `command` must be at `.centimetres` scale (a
    /// `.normalized` command throws `UEMACoreError.wrongScale`); `gaze` nil means the pair
    /// read from the command's eyes' row (`Command.gazeAngles`); `fingerGoals` nil means the
    /// finger rows are decoded from their quaternions like every other row.
    func decode(_ command: Command, mask: HandleMask, gaze: GazeAngles?, fingerGoals: FingerJointGoals?) throws -> MotorControls
}

// MARK: - The eyes

/// One of the five eye cameras the vision sense renders (cameras.json of WI-14; plan row
/// WI-29). The raw value is the camera's name in the scene XML.
public enum EyeCamera: String, CaseIterable, Codable, Hashable, Sendable {
    case eyeLeft = "eye_left"
    case eyeRight = "eye_right"
    case foveaLeft = "fovea_left"
    case foveaRight = "fovea_right"
    case ringLeft = "ring_left"

    /// The square image size in pixels: 256 for the eyes and the ring, 288 for the foveas.
    public var pixelSize: Int {
        switch self {
        case .eyeLeft, .eyeRight, .ringLeft:
            return 256
        case .foveaLeft, .foveaRight:
            return 288
        }
    }

    /// The vertical field of view in degrees: 60 for the eyes, 24 for the foveas, 150 for
    /// the ring.
    public var fieldOfViewDegrees: Double {
        switch self {
        case .eyeLeft, .eyeRight:
            return 60
        case .foveaLeft, .foveaRight:
            return 24
        case .ringLeft:
            return 150
        }
    }

    /// The MuJoCo body the camera is attached to: left_eye or right_eye.
    public var eyeBodyName: String {
        switch self {
        case .eyeLeft, .foveaLeft, .ringLeft:
            return "left_eye"
        case .eyeRight, .foveaRight:
            return "right_eye"
        }
    }

    /// The camera's name in the scene XML.
    public var mujocoName: String { rawValue }
}

/// A camera's pose in the world: position (3, metres) and rotation matrix (9, row-major),
/// as cameras.json records them (`pos_m`, `xmat_row_major`).
public struct CameraPose: Equatable, Hashable, Codable, Sendable {
    /// The camera position, 3 numbers, metres, world frame.
    public let position: [Double]
    /// The camera rotation matrix, 9 numbers, row-major.
    public let rotation: [Double]

    /// A pose; a position of other than 3 numbers or a rotation of other than 9 throws.
    public init(position: [Double], rotation: [Double]) throws {
        guard position.count == 3 else {
            throw UEMACoreError.wrongCount(type: "CameraPose.position", expected: 3, found: position.count)
        }
        guard rotation.count == 9 else {
            throw UEMACoreError.wrongCount(type: "CameraPose.rotation", expected: 9, found: rotation.count)
        }
        self.position = position
        self.rotation = rotation
    }

    private enum CodingKeys: String, CodingKey {
        case position
        case rotation
    }

    /// Decodes through the count checks.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            position: container.decode([Double].self, forKey: .position),
            rotation: container.decode([Double].self, forKey: .rotation))
    }
}

/// One geom of the scene as MuJoCo's mjvGeom gives it after mjv_updateScene (MuJoCo API
/// reference, quoted in this file's header): `type` the mjtGeom value, `size` the 3 size
/// parameters, `position` the 3 Cartesian coordinates, `rotation` the 9-number orientation
/// matrix, `rgba` the colour and transparency. GeomMirror (plan row WI-29) maps each to a
/// SceneKit node.
public struct SceneGeom: Equatable, Hashable, Codable, Sendable {
    /// The mjtGeom value (mjvGeom.type), such as 2 for a sphere.
    public let type: Int
    /// The 3 size parameters (mjvGeom.size), metres.
    public let size: [Float]
    /// The 3 Cartesian coordinates (mjvGeom.pos), metres, world frame.
    public let position: [Float]
    /// The 9-number orientation matrix (mjvGeom.mat), row-major.
    public let rotation: [Float]
    /// Red, green, blue and opacity, each 0 to 1 (mjvGeom.rgba).
    public let rgba: [Float]

    /// A geom; counts other than size 3, position 3, rotation 9, rgba 4 throw.
    public init(type: Int, size: [Float], position: [Float], rotation: [Float], rgba: [Float]) throws {
        guard size.count == 3 else {
            throw UEMACoreError.wrongCount(type: "SceneGeom.size", expected: 3, found: size.count)
        }
        guard position.count == 3 else {
            throw UEMACoreError.wrongCount(type: "SceneGeom.position", expected: 3, found: position.count)
        }
        guard rotation.count == 9 else {
            throw UEMACoreError.wrongCount(type: "SceneGeom.rotation", expected: 9, found: rotation.count)
        }
        guard rgba.count == 4 else {
            throw UEMACoreError.wrongCount(type: "SceneGeom.rgba", expected: 4, found: rgba.count)
        }
        self.type = type
        self.size = size
        self.position = position
        self.rotation = rotation
        self.rgba = rgba
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case size
        case position
        case rotation
        case rgba
    }

    /// Decodes through the count checks.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            type: container.decode(Int.self, forKey: .type),
            size: container.decode([Float].self, forKey: .size),
            position: container.decode([Float].self, forKey: .position),
            rotation: container.decode([Float].self, forKey: .rotation),
            rgba: container.decode([Float].self, forKey: .rgba))
    }
}

/// One rendered eye image: RGB 8-bit, rows top-down, columns left-to-right (cameras.json
/// image_format), `rgb.count` = width x height x 3.
public struct EyeImage: Equatable, Hashable, Sendable {
    /// The camera that rendered the image.
    public let camera: EyeCamera
    /// Columns, in pixels.
    public let width: Int
    /// Rows, in pixels.
    public let height: Int
    /// width x height x 3 numbers, 0 to 255: rows top-down, then columns, then red, green, blue.
    public let rgb: [UInt8]

    /// An image; a negative width or height, or an `rgb` count other than width x height x 3,
    /// throws `UEMACoreError.wrongCount` (the expected count is -1 when width x height x 3 is
    /// not a valid count).
    public init(camera: EyeCamera, width: Int, height: Int, rgb: [UInt8]) throws {
        let (pixels, pixelsOverflow) = width.multipliedReportingOverflow(by: height)
        let (expected, valuesOverflow) = pixels.multipliedReportingOverflow(by: 3)
        let isValidSize = width >= 0 && height >= 0 && !pixelsOverflow && !valuesOverflow
        guard isValidSize, rgb.count == expected else {
            throw UEMACoreError.wrongCount(
                type: "EyeImage.rgb", expected: isValidSize ? expected : -1, found: rgb.count)
        }
        self.camera = camera
        self.width = width
        self.height = height
        self.rgb = rgb
    }
}

/// The geom list and camera poses of the current state, the inputs of an eye render
/// (mjv_updateScene fills the list, plan row WI-29). Implemented beside the physics in
/// UEMAPhysics; a test feeds a renderer synthetic geoms through a fake.
public protocol SceneGeomReading: AnyObject {
    /// Every visible geom of the current state.
    func sceneGeoms() throws -> [SceneGeom]

    /// The current pose of one eye camera.
    func cameraPose(_ camera: EyeCamera) throws -> CameraPose
}

/// Renders one eye camera offscreen into an RGB image (plan row WI-29; specification
/// REQ-B-6): a pure function of the camera, its pose and the geom list, so that the render
/// of a fixture state needs no physics. Class-bound: a renderer keeps its device and nodes.
public protocol EyeRendering: AnyObject {
    /// The image of `camera` at `camera.pixelSize` square pixels and
    /// `camera.fieldOfViewDegrees`.
    func render(camera: EyeCamera, pose: CameraPose, geoms: [SceneGeom]) throws -> EyeImage
}
