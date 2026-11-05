// ProtocolsTests.swift: the tests of Protocols.swift (work item WI-20): the shape and count
// checks of the value types that cross the protocols, the five eye cameras, the default step
// methods of Physics (10 physics steps and 5 control writes per model step), the default
// planning condition of Planning (20 goal ring entries and a model step count divisible by
// 10), and one fake of every protocol, so that a requirement change in Protocols.swift shows
// up here as a compile error before it shows up in a later module.
// Not compiled on the machine that wrote it (no Swift toolchain on Windows).
//
// Sources of the numbers below: cameras.json of WI-14 (camera sizes, fields of view, eye
// bodies; Protocols.swift header); weight_manifest.json of WI-12 (the "contact_head.ctx2.bias"
// row, apple/UEMAKit/Sources/UEMAModel/Resources/weight_manifest.json); planner_loss.md
// section 3 (21 evaluations, 20 updates, Adam step size 0.1); command_decoder.md sections 2
// and 5 point C (the motor contract string); MuJoCo mjtGeom (type 2 is a sphere).

import Foundation
import XCTest
import UEMACore

// MARK: - Fakes

/// A test error for the fakes.
private enum FakeError: Error, Equatable {
    case notLoaded
    case noSuchFile(String)
}

/// A Physics fake that keeps a log of its calls and advances its time by 0.005 s per
/// physics step. It implements only the requirements without a default, so the two step
/// methods under test are the protocol extension's.
private final class FakePhysics: Physics {
    var sizes = PhysicsSizes(nq: 146, nv: 136, nu: 92, nbody: 76, njnt: 97, nsensordata: 56, ngeom: 0, ncam: 5)
    var time: Double = 0
    var qpos: [Double] = Array(repeating: 0, count: 146)
    var qvel: [Double] = Array(repeating: 0, count: 136)
    var sensordata: [Double] = Array(repeating: 0, count: 56)
    var xpos: [Double] = Array(repeating: 0, count: 76 * 3)
    var xquat: [Double] = Array(repeating: 0, count: 76 * 4)
    var xmat: [Double] = Array(repeating: 0, count: 76 * 9)
    var xaxis: [Double] = Array(repeating: 0, count: 97 * 3)
    var xanchor: [Double] = Array(repeating: 0, count: 97 * 3)
    var actuatorForce: [Double] = Array(repeating: 0, count: 92)
    var control: [Double] = Array(repeating: 0, count: 92)
    var contacts: [Contact] = []
    var isLoaded = false
    var log: [String] = []

    func id(of kind: MuJoCoObjectKind, named name: String) -> Int? {
        kind == .body && name == "left_eye" ? 7 : nil
    }

    func bodyVelocity(bodyId: Int) -> [Double] {
        Array(repeating: 0, count: 6)
    }

    func load(sceneXML url: URL) throws {
        isLoaded = true
        log.append("load")
    }

    func reset() throws {
        time = 0
        log.append("reset")
    }

    func setState(qpos: [Double], qvel: [Double]) throws {
        guard qpos.count == sizes.nq else {
            throw UEMACoreError.wrongCount(type: "qpos", expected: sizes.nq, found: qpos.count)
        }
        guard qvel.count == sizes.nv else {
            throw UEMACoreError.wrongCount(type: "qvel", expected: sizes.nv, found: qvel.count)
        }
        self.qpos = qpos
        self.qvel = qvel
    }

    func write(controls: MotorControls) throws {
        guard isLoaded else { throw FakeError.notLoaded }
        control = controls.clipped().values
        log.append("write")
    }

    func stepPhysicsStep() throws {
        guard isLoaded else { throw FakeError.notLoaded }
        time += ModelStep.physicsTimestepSeconds
        log.append("step")
    }
}

/// A SenseReader fake that implements only `sense` and `read`, so the observe and reset
/// methods are the protocol extension's (which do nothing).
private final class FakeVestibularReader: SenseReader {
    let sense: SenseName = .vestibular
    var reads = 0

    func read(_ physics: any PhysicsReading, at step: ModelStep) throws -> SenseReading {
        reads += 1
        return SenseReading(sense: sense, primary: ShapedArray(zerosOfShape: [29]), isFresh: step.count > 0)
    }
}

/// A Tokenizer fake that drops the last number of a [29] reading: it tests the protocol's
/// shape, not the vestibular arithmetic (that is WI-30's).
private struct FakeVestibularTokenizer: Tokenizer {
    let sense: SenseName = .vestibular

    func tokenize(_ reading: SenseReading) throws -> EncoderInput {
        guard reading.sense == sense else {
            throw UEMACoreError.missingSense(sense)
        }
        guard reading.primary.shape == [29] else {
            throw UEMACoreError.wrongShape(type: "vestibular reading", expected: [29], found: reading.primary.shape)
        }
        let primary = try ShapedArray(shape: [28], values: Array(reading.primary.values.prefix(28)))
        return EncoderInput(sense: sense, primary: primary, isFresh: reading.isFresh)
    }
}

/// A Planning fake whose ring counts recorded model steps up to 200; `canPlan` is the
/// protocol extension's.
private final class FakePlanner: Planning {
    private(set) var goalRingCount = 0

    func record(tokens: TokenSet, wellness: Float, at step: ModelStep) {
        goalRingCount = min(goalRingCount + 1, ModelStep.goalRingLength)
    }

    func plan(from command: Command, at step: ModelStep) throws -> PlanResult {
        try PlanResult(
            command: command, scores: [2, 1], selectedIteration: 1,
            goalStep: step, goalWellness: 0.5, fingerGoals: nil)
    }

    func reset() {
        goalRingCount = 0
    }
}

/// A WeightsLoading fake whose arrays are a dictionary of float lists.
private struct FakeWeightsLoader: WeightsLoading {
    typealias Weights = [String: [Float]]

    func load(from url: URL) throws -> (weights: [String: [Float]], report: WeightsReport) {
        let weights: [String: [Float]] = ["contact_head.ctx2.bias": Array(repeating: 0, count: 256)]
        let report = WeightsReport(
            fileURL: url, loadedKeys: ["contact_head.ctx2.bias"], skippedKeys: ["optimizer.state"],
            loadedElements: 256)
        return (weights, report)
    }
}

/// Reads the loaded element count through the protocol only (generic over the conformer).
private func loadedElements<Loader: WeightsLoading>(with loader: Loader, url: URL) throws -> Int {
    try loader.load(from: url).report.loadedElements
}

/// A ForwardModeling fake that checks for six inputs and returns zero tokens.
private final class FakeForwardModel: ForwardModeling {
    let contextLength = ModelStep.contextLength
    private(set) var filledContextSteps = 0

    func step(inputs: [SenseName: EncoderInput], previousExecutedCommand: Command?, at step: ModelStep) throws -> ModelOutput {
        for sense in SenseName.active where inputs[sense] == nil {
            throw UEMACoreError.missingSense(sense)
        }
        if let previous = previousExecutedCommand, previous.scale != .centimetres {
            throw UEMACoreError.wrongScale(expected: .centimetres, found: previous.scale)
        }
        filledContextSteps = min(filledContextSteps + 1, contextLength)
        var tokens: TokenSet = [:]
        for sense in SenseName.active {
            tokens[sense] = Token.zero
        }
        return ModelOutput(
            step: step, tokens: tokens, predictedTokens: tokens,
            command: Command.zero(scale: .normalized), wellness: 0)
    }

    func reset() {
        filledContextSteps = 0
    }
}

/// A CommandDecoding fake that copies the wrist_L x position (cm) into control 0, clipped.
private final class FakeDecoder: CommandDecoding {
    let motorContract = "ancestor_arm_axes_signed_limits_v3+finger_joint_servo_v1"

    func decode(_ command: Command, mask: HandleMask, gaze: GazeAngles?, fingerGoals: FingerJointGoals?) throws -> MotorControls {
        guard command.scale == .centimetres else {
            throw UEMACoreError.wrongScale(expected: .centimetres, found: command.scale)
        }
        var values = Array(repeating: 0.0, count: MotorControls.count)
        values[0] = Double(command[.wristL, .positionX])
        return try MotorControls(values: values).clipped()
    }
}

/// An EyeRendering fake that fills the image with the geom count.
private final class FakeEyeRenderer: EyeRendering {
    func render(camera: EyeCamera, pose: CameraPose, geoms: [SceneGeom]) throws -> EyeImage {
        let side = camera.pixelSize
        return try EyeImage(
            camera: camera, width: side, height: side,
            rgb: Array(repeating: UInt8(min(geoms.count, 255)), count: side * side * 3))
    }
}

/// A SceneGeomReading fake with one red sphere (mjtGeom 2) and every camera at the origin.
private final class FakeGeomReader: SceneGeomReading {
    func sceneGeoms() throws -> [SceneGeom] {
        let sphere = try SceneGeom(
            type: 2, size: [0.1, 0, 0], position: [0, 0, 0.5],
            rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], rgba: [1, 0, 0, 1])
        return [sphere]
    }

    func cameraPose(_ camera: EyeCamera) throws -> CameraPose {
        try CameraPose(position: [0, 0, 0.3], rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1])
    }
}

/// A SceneLoading fake over a folder URL.
private struct FakeScene: SceneLoading {
    let folder: URL

    var sceneXMLURL: URL { folder.appendingPathComponent("scene.xml") }

    var skinLayoutURL: URL { folder.appendingPathComponent("skin_layout.json") }

    func url(forFile name: String) throws -> URL {
        guard name == "scene.xml" || name == "skin_layout.json" else {
            throw FakeError.noSuchFile(name)
        }
        return folder.appendingPathComponent(name)
    }
}

// MARK: - Tests

final class ProtocolsTests: XCTestCase {
    private let identity: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]

    /// A shaped array holds exactly the product of its shape; a scalar has shape [].
    func testShapedArrayChecksItsShape() throws {
        let array = try ShapedArray(shape: [26, 12], values: Array(repeating: 1, count: 312))
        XCTAssertEqual(array.count, 312)
        XCTAssertEqual(array.rank, 2)
        let scalar = try ShapedArray(shape: [], values: [3])
        XCTAssertEqual(scalar.rank, 0)
        XCTAssertEqual(scalar.count, 1)
        XCTAssertThrowsError(try ShapedArray(shape: [26, 12], values: [1, 2])) { error in
            XCTAssertEqual(error as? UEMACoreError, .invalidShape(type: "ShapedArray", shape: [26, 12], valueCount: 2))
        }
        XCTAssertThrowsError(try ShapedArray(shape: [-1, -2], values: [1, 2]))
        // A shape whose product overflows Int throws instead of stopping the process.
        XCTAssertThrowsError(try ShapedArray(shape: [Int.max, 2], values: []))
        let empty = try ShapedArray(shape: [0, 768], values: [])
        XCTAssertEqual(empty.count, 0)
        let zeros = ShapedArray(zerosOfShape: [32, 19])
        XCTAssertEqual(zeros.count, 608)
        XCTAssertTrue(zeros.values.allSatisfy { $0 == 0 })
        let data = try JSONEncoder().encode(array)
        XCTAssertEqual(try JSONDecoder().decode(ShapedArray.self, from: data), array)
        let bad = Data(#"{"shape":[2,2],"values":[1,2,3]}"#.utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(ShapedArray.self, from: bad))
    }

    /// A reading and an encoder input carry their sense, arrays and freshness flag.
    func testSenseReadingAndEncoderInputCarryFreshness() throws {
        let patches = ShapedArray(zerosOfShape: [4, 768])
        let angles = ShapedArray(zerosOfShape: [4, 4])
        let input = EncoderInput(sense: .vision, primary: patches, auxiliary: angles, isFresh: false)
        XCTAssertEqual(input.sense, .vision)
        XCTAssertEqual(input.primary.shape, [4, 768])
        XCTAssertEqual(input.auxiliary?.shape, [4, 4])
        XCTAssertFalse(input.isFresh)
        let reading = SenseReading(sense: .audio, primary: ShapedArray(zerosOfShape: [256, 2]))
        XCTAssertNil(reading.auxiliary)
        XCTAssertTrue(reading.isFresh)
    }

    /// Contact, camera pose, scene geom and eye image refuse arrays of the wrong count.
    func testCountChecksOfContactPoseGeomAndImage() throws {
        let contact = try Contact(
            geomIds: [3, -1], distance: -0.001, position: [0, 0, 0.1], frame: identity,
            force: [1, 0, 0, 0, 0, 0])
        XCTAssertEqual(contact.force.count, 6)
        XCTAssertThrowsError(try Contact(geomIds: [3], distance: 0, position: [0, 0, 0], frame: identity, force: [0, 0, 0, 0, 0, 0]))
        XCTAssertThrowsError(try Contact(geomIds: [1, 2], distance: 0, position: [0, 0], frame: identity, force: [0, 0, 0, 0, 0, 0]))
        XCTAssertThrowsError(try Contact(geomIds: [1, 2], distance: 0, position: [0, 0, 0], frame: [1], force: [0, 0, 0, 0, 0, 0]))
        XCTAssertThrowsError(try Contact(geomIds: [1, 2], distance: 0, position: [0, 0, 0], frame: identity, force: [0, 0, 0]))
        XCTAssertThrowsError(try CameraPose(position: [0, 0], rotation: identity))
        XCTAssertThrowsError(try CameraPose(position: [0, 0, 0], rotation: [1, 0, 0]))
        XCTAssertThrowsError(try SceneGeom(type: 2, size: [1], position: [0, 0, 0], rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], rgba: [1, 1, 1, 1]))
        XCTAssertThrowsError(try SceneGeom(type: 2, size: [1, 0, 0], position: [0, 0, 0], rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], rgba: [1, 1, 1]))
        XCTAssertThrowsError(try EyeImage(camera: .eyeLeft, width: 2, height: 2, rgb: [0, 0, 0]))
        let image = try EyeImage(camera: .eyeLeft, width: 2, height: 2, rgb: Array(repeating: 9, count: 12))
        XCTAssertEqual(image.rgb.count, 12)
        XCTAssertThrowsError(try EyeImage(camera: .eyeLeft, width: -2, height: -2, rgb: Array(repeating: 9, count: 12))) { error in
            XCTAssertEqual(error as? UEMACoreError, .wrongCount(type: "EyeImage.rgb", expected: -1, found: 12))
        }
        XCTAssertThrowsError(try EyeImage(camera: .eyeLeft, width: Int.max, height: 2, rgb: []))
        let data = try JSONEncoder().encode(contact)
        XCTAssertEqual(try JSONDecoder().decode(Contact.self, from: data), contact)
        let badContact = Data(#"{"geomIds":[1],"distance":0,"position":[0,0,0],"frame":[1,0,0,0,1,0,0,0,1],"force":[0,0,0,0,0,0]}"#.utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(Contact.self, from: badContact))
    }

    /// The five eye cameras match cameras.json: sizes 256 and 288 px, fields of view 60, 24
    /// and 150 degrees, attached to left_eye or right_eye.
    func testEyeCamerasMatchTheFixture() {
        XCTAssertEqual(EyeCamera.allCases.count, 5)
        XCTAssertEqual(EyeCamera.eyeLeft.mujocoName, "eye_left")
        XCTAssertEqual(EyeCamera.ringLeft.rawValue, "ring_left")
        XCTAssertEqual(EyeCamera.eyeRight.pixelSize, 256)
        XCTAssertEqual(EyeCamera.foveaLeft.pixelSize, 288)
        XCTAssertEqual(EyeCamera.ringLeft.pixelSize, 256)
        XCTAssertEqual(EyeCamera.eyeLeft.fieldOfViewDegrees, 60, accuracy: 1e-12)
        XCTAssertEqual(EyeCamera.foveaRight.fieldOfViewDegrees, 24, accuracy: 1e-12)
        XCTAssertEqual(EyeCamera.ringLeft.fieldOfViewDegrees, 150, accuracy: 1e-12)
        XCTAssertEqual(EyeCamera.ringLeft.eyeBodyName, "left_eye")
        XCTAssertEqual(EyeCamera.foveaRight.eyeBodyName, "right_eye")
        XCTAssertEqual(EyeCamera.allCases.filter { $0.eyeBodyName == "left_eye" }.count, 3)
    }

    /// The default stepModelStep runs 5 env steps, each one control write then 2 physics
    /// steps with the callback after each: 10 physics steps, 10 callbacks, 0.050 s.
    func testPhysicsDefaultModelStepRunsTenPhysicsSteps() throws {
        let physics = FakePhysics()
        try physics.load(sceneXML: URL(fileURLWithPath: "/scene.xml"))
        XCTAssertTrue(physics.isLoaded)
        physics.log.removeAll()
        var callbacks = 0
        var callbackTimes: [Double] = []
        var values = Array(repeating: 0.0, count: MotorControls.count)
        values[0] = 2.5
        try physics.stepModelStep(controls: MotorControls(values: values)) { reading in
            callbacks += 1
            callbackTimes.append(reading.time)
        }
        XCTAssertEqual(callbacks, ModelStep.physicsStepsPerModelStep)
        XCTAssertEqual(physics.log.filter { $0 == "step" }.count, 10)
        XCTAssertEqual(physics.log.filter { $0 == "write" }.count, 5)
        XCTAssertEqual(Array(physics.log.prefix(3)), ["write", "step", "step"])
        XCTAssertEqual(physics.time, ModelStep.seconds, accuracy: 1e-12)
        XCTAssertEqual(callbackTimes.first ?? -1, 0.005, accuracy: 1e-12)
        XCTAssertEqual(physics.control[0], 1, "the fake clips 2.5 to 1, as WI-25 requires")

        let existential: any Physics = physics
        try existential.stepEnvStep(controls: MotorControls.zero) { _ in }
        XCTAssertEqual(physics.time, ModelStep.seconds + ModelStep.envStepSeconds, accuracy: 1e-12)
        XCTAssertEqual(physics.id(of: .body, named: "left_eye"), 7)
        XCTAssertNil(physics.id(of: .camera, named: "left_eye"))
        XCTAssertThrowsError(try physics.setState(qpos: [0], qvel: []))

        let unloaded = FakePhysics()
        XCTAssertThrowsError(try unloaded.stepModelStep(controls: MotorControls.zero) { _ in }) { error in
            XCTAssertEqual(error as? FakeError, .notLoaded)
        }
    }

    /// The default observe and reset methods of SenseReader do nothing; a tokenizer turns the
    /// reading into an encoder input and refuses a reading of another sense.
    func testSenseReaderDefaultsAndTokenizer() throws {
        let physics = FakePhysics()
        let reader = FakeVestibularReader()
        try reader.observePhysicsStep(physics)
        try reader.observeEnvStep(physics)
        reader.reset()
        XCTAssertEqual(reader.reads, 0)
        let reading = try reader.read(physics, at: ModelStep(count: 3))
        XCTAssertEqual(reader.reads, 1)
        XCTAssertEqual(reading.sense, .vestibular)
        XCTAssertEqual(reading.primary.shape, [29])
        XCTAssertTrue(reading.isFresh)

        let tokenizer = FakeVestibularTokenizer()
        let input = try tokenizer.tokenize(reading)
        XCTAssertEqual(input.sense, .vestibular)
        XCTAssertEqual(input.primary.shape, [28])
        let other = SenseReading(sense: .audio, primary: ShapedArray(zerosOfShape: [256, 2]))
        XCTAssertThrowsError(try tokenizer.tokenize(other)) { error in
            XCTAssertEqual(error as? UEMACoreError, .missingSense(.vestibular))
        }
    }

    /// The default canPlan needs at least 20 goal ring entries and a count divisible by 10.
    func testPlanningDefaultCondition() throws {
        let planner = FakePlanner()
        XCTAssertFalse(planner.canPlan(at: ModelStep(count: 10)))
        for count in 0 ..< 19 {
            planner.record(tokens: [:], wellness: 0, at: ModelStep(count: count))
        }
        XCTAssertEqual(planner.goalRingCount, 19)
        XCTAssertFalse(planner.canPlan(at: ModelStep(count: 20)))
        planner.record(tokens: [:], wellness: 0, at: ModelStep(count: 19))
        XCTAssertTrue(planner.canPlan(at: ModelStep(count: 20)))
        XCTAssertFalse(planner.canPlan(at: ModelStep(count: 21)))
        for count in 20 ..< 400 {
            planner.record(tokens: [:], wellness: 0, at: ModelStep(count: count))
        }
        XCTAssertEqual(planner.goalRingCount, 200)
        let result = try planner.plan(from: Command.zero(scale: .normalized), at: ModelStep(count: 400))
        XCTAssertEqual(result.selectedScore, 1)
        planner.reset()
        XCTAssertEqual(planner.goalRingCount, 0)
        XCTAssertFalse(planner.canPlan(at: ModelStep(count: 400)))
    }

    /// A plan result holds 1 to 21 scores, a selected index inside them and a command at the
    /// normalized scale; it survives an encode and decode.
    func testPlanResultChecks() throws {
        XCTAssertEqual(PlanResult.evaluationsPerEvent, 21)
        XCTAssertEqual(PlanResult.updatesPerEvent, 20)
        XCTAssertEqual(PlanResult.evaluationsPerEvent, PlanResult.updatesPerEvent + 1)
        XCTAssertEqual(PlanResult.learningRate, 0.1, accuracy: 1e-7)
        XCTAssertEqual(PlanResult.strengthRegularisation, 0.01, accuracy: 1e-7)
        let command = Command.zero(scale: .normalized)
        let scores = (0 ..< 21).map { Float(21 - $0) }
        let goals = try FingerJointGoals(angles: Array(repeating: 0.2, count: 44))
        let result = try PlanResult(
            command: command, scores: scores, selectedIteration: 20,
            goalStep: ModelStep(count: 130), goalWellness: 0.75, fingerGoals: goals)
        XCTAssertEqual(result.selectedScore, 1)
        let data = try JSONEncoder().encode(result)
        XCTAssertEqual(try JSONDecoder().decode(PlanResult.self, from: data), result)
        XCTAssertThrowsError(try PlanResult(
            command: Command.zero(scale: .centimetres), scores: scores, selectedIteration: 0,
            goalStep: .start, goalWellness: 0, fingerGoals: nil)) { error in
            XCTAssertEqual(error as? UEMACoreError, .wrongScale(expected: .normalized, found: .centimetres))
        }
        XCTAssertThrowsError(try PlanResult(
            command: command, scores: [], selectedIteration: 0, goalStep: .start, goalWellness: 0, fingerGoals: nil))
        XCTAssertThrowsError(try PlanResult(
            command: command, scores: Array(repeating: 0, count: 22), selectedIteration: 0,
            goalStep: .start, goalWellness: 0, fingerGoals: nil))
        XCTAssertThrowsError(try PlanResult(
            command: command, scores: scores, selectedIteration: 21,
            goalStep: .start, goalWellness: 0, fingerGoals: nil)) { error in
            XCTAssertEqual(error as? UEMACoreError, .indexOutOfRange(type: "PlanResult.selectedIteration", index: 21, count: 21))
        }
    }

    /// A manifest row of weight_manifest.json decodes with its "inference_set" key, and a
    /// loader is usable through the protocol alone.
    func testWeightsManifestEntryAndLoader() throws {
        let row = Data(#"{"key": "contact_head.ctx2.bias", "shape": [256], "dtype": "F32", "elements": 256, "bytes": 1024, "component": "contact_head", "kind": "parameter", "inference_set": false}"#.utf8)
        let entry = try JSONDecoder().decode(WeightsManifestEntry.self, from: row)
        XCTAssertEqual(entry.key, "contact_head.ctx2.bias")
        XCTAssertEqual(entry.shape, [256])
        XCTAssertEqual(entry.bytes, entry.elements * 4)
        XCTAssertFalse(entry.inferenceSet)
        let url = URL(fileURLWithPath: "/weights/model.safetensors")
        XCTAssertEqual(try loadedElements(with: FakeWeightsLoader(), url: url), 256)
        let (weights, report) = try FakeWeightsLoader().load(from: url)
        XCTAssertEqual(weights["contact_head.ctx2.bias"]?.count, 256)
        XCTAssertEqual(report.fileURL, url)
        XCTAssertEqual(report.skippedKeys, ["optimizer.state"])
    }

    /// One fake each of ForwardModeling, CommandDecoding, EyeRendering, SceneGeomReading and
    /// SceneLoading, driven in the session's sequence for one model step.
    func testModelDecoderRendererAndSceneFakes() throws {
        let model = FakeForwardModel()
        XCTAssertEqual(model.contextLength, 128)
        var inputs: [SenseName: EncoderInput] = [:]
        for sense in SenseName.active {
            inputs[sense] = EncoderInput(sense: sense, primary: ShapedArray(zerosOfShape: [1]))
        }
        let output = try model.step(inputs: inputs, previousExecutedCommand: nil, at: .start)
        XCTAssertEqual(model.filledContextSteps, 1)
        XCTAssertTrue(output.tokens.missingActiveSenses.isEmpty)
        XCTAssertEqual(try output.predictedTokens.activeSequence().count, 6)
        XCTAssertEqual(output.command.scale, .normalized)
        inputs[.motor] = nil
        XCTAssertThrowsError(try model.step(inputs: inputs, previousExecutedCommand: nil, at: ModelStep(count: 1))) { error in
            XCTAssertEqual(error as? UEMACoreError, .missingSense(.motor))
        }
        model.reset()
        XCTAssertEqual(model.filledContextSteps, 0)

        let decoder = FakeDecoder()
        XCTAssertEqual(decoder.motorContract, "ancestor_arm_axes_signed_limits_v3+finger_joint_servo_v1")
        var executed = Command.zero(scale: .centimetres)
        executed[.wristL, .positionX] = 30
        let controls = try decoder.decode(executed, mask: .allTrue, gaze: nil, fingerGoals: nil)
        XCTAssertEqual(controls.values.count, 92)
        XCTAssertEqual(controls.values[0], 1)
        XCTAssertTrue(controls.isWithinRange)
        XCTAssertThrowsError(try decoder.decode(output.command, mask: .allTrue, gaze: .straightAhead, fingerGoals: nil)) { error in
            XCTAssertEqual(error as? UEMACoreError, .wrongScale(expected: .centimetres, found: .normalized))
        }

        let geomReader = FakeGeomReader()
        let renderer = FakeEyeRenderer()
        let geoms = try geomReader.sceneGeoms()
        for camera in EyeCamera.allCases {
            let image = try renderer.render(camera: camera, pose: geomReader.cameraPose(camera), geoms: geoms)
            XCTAssertEqual(image.width, camera.pixelSize)
            XCTAssertEqual(image.height, camera.pixelSize)
            XCTAssertEqual(image.rgb.count, camera.pixelSize * camera.pixelSize * 3)
            XCTAssertEqual(image.rgb.first, 1)
        }

        let scene = FakeScene(folder: URL(fileURLWithPath: "/bundle", isDirectory: true))
        XCTAssertEqual(scene.sceneXMLURL.lastPathComponent, "scene.xml")
        XCTAssertEqual(try scene.url(forFile: "skin_layout.json"), scene.skinLayoutURL)
        XCTAssertThrowsError(try scene.url(forFile: "absent.xml")) { error in
            XCTAssertEqual(error as? FakeError, .noSuchFile("absent.xml"))
        }
    }
}
