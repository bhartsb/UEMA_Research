// RunViewModelTests.swift: the tests of work item WI-32 (apple/UEMAMind/ViewModels/), run
// on fakes of every UEMACore protocol, so no MuJoCo, MLX or SceneKit code runs. Target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html. The test bundles
// UEMAMindTests-macOS and UEMAMindTests-iOS compile this folder (apple/project.yml) and
// are hosted by the app they test.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). The "First build on a Mac" checklist is in
// RunViewModel.swift's header (step 3 runs these 15 tests).
//
// The acceptance line of WI-32 and the test that checks each part of it:
//   "with fakes, 10 model steps call the planner once"  testTenModelStepsCallThePlannerOnce
//   "a fake critical thermal state skips it"            testCriticalThermalStateSkipsThePlanningEvent
//   "a background notice stops the loop"                testBackgroundNoticeStopsTheLoop

import Foundation
import SwiftUI
import UEMACore
import XCTest

#if os(macOS)
@testable import UEMAMind_macOS
#else
@testable import UEMAMind_iOS
#endif

// MARK: - Stubs of the UEMACore protocols (named Stub*, apart from the Fake* types of UEMAMindApp.swift)

/// Physics that only counts: every physics step adds 0.005 s and 1 to `physicsSteps`.
final class StubPhysics: Physics {
    var physicsSteps = 0
    var writtenControls: [MotorControls] = []
    var resets = 0

    let sizes = PhysicsSizes(nq: 1, nv: 1, nu: MotorControls.count, nbody: 1, njnt: 1, nsensordata: 0, ngeom: 0, ncam: 0)
    var time: Double = 0
    var qpos: [Double] = [0]
    var qvel: [Double] = [0]
    var sensordata: [Double] = []
    var xpos: [Double] = [0, 0, 0]
    var xquat: [Double] = [1, 0, 0, 0]
    var xmat: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    var xaxis: [Double] = [0, 0, 1]
    var xanchor: [Double] = [0, 0, 0]
    var actuatorForce: [Double] = Array(repeating: 0, count: MotorControls.count)
    var control: [Double] = Array(repeating: 0, count: MotorControls.count)
    var contacts: [Contact] = []
    var isLoaded = true

    func load(sceneXML url: URL) throws {
        isLoaded = true
    }

    func reset() throws {
        resets += 1
        time = 0
    }

    func setState(qpos: [Double], qvel: [Double]) throws {
        self.qpos = qpos
        self.qvel = qvel
    }

    func write(controls: MotorControls) throws {
        writtenControls.append(controls)
        control = controls.values
    }

    func stepPhysicsStep() throws {
        physicsSteps += 1
        time += ModelStep.physicsTimestepSeconds
    }

    func id(of kind: MuJoCoObjectKind, named name: String) -> Int? {
        nil
    }

    func bodyVelocity(bodyId: Int) -> [Double] {
        [0, 0, 0, 0, 0, 0]
    }
}

/// A reader that counts its calls and returns [model step count, -1].
final class StubSenseReader: SenseReader {
    let sense: SenseName
    var physicsStepsSeen = 0
    var envStepsSeen = 0
    var reads = 0
    var resets = 0

    init(sense: SenseName) {
        self.sense = sense
    }

    func observePhysicsStep(_ physics: any PhysicsReading) throws {
        physicsStepsSeen += 1
    }

    func observeEnvStep(_ physics: any PhysicsReading) throws {
        envStepsSeen += 1
    }

    func read(_ physics: any PhysicsReading, at step: ModelStep) throws -> SenseReading {
        reads += 1
        return SenseReading(sense: sense, primary: try ShapedArray(shape: [2], values: [Float(step.count), -1]))
    }

    func reset() {
        resets += 1
    }
}

/// A tokenizer that copies the reading into the encoder input.
struct StubTokenizer: Tokenizer {
    let sense: SenseName

    func tokenize(_ reading: SenseReading) throws -> EncoderInput {
        EncoderInput(sense: reading.sense, primary: reading.primary, auxiliary: reading.auxiliary, isFresh: reading.isFresh)
    }
}

/// A forward model whose tokens at model step n are all n, whose predicted tokens are all
/// n + 1, whose command has wrist_L x = n, and whose wellness is n. With
/// `gazeAzimuthPerModelStep` set to a, the command's eyes' row holds azimuth a x n,
/// elevation 0 (Command.setGaze).
final class StubForwardModel: ForwardModeling {
    let contextLength = ModelStep.contextLength
    private(set) var filledContextSteps = 0
    var previousExecuted: [Command?] = []
    var gazeAzimuthPerModelStep: Double? = nil

    func step(inputs: [SenseName: EncoderInput], previousExecutedCommand: Command?, at step: ModelStep) throws -> ModelOutput {
        previousExecuted.append(previousExecutedCommand)
        filledContextSteps = min(filledContextSteps + 1, contextLength)
        var tokens: TokenSet = [:]
        var predicted: TokenSet = [:]
        for sense in SenseName.active {
            tokens[sense] = Token(repeating: Float(step.count))
            predicted[sense] = Token(repeating: Float(step.count + 1))
        }
        var command = Command.zero(scale: .normalized)
        command[.wristL, .positionX] = Float(step.count)
        if let azimuthStep = gazeAzimuthPerModelStep {
            command.setGaze(GazeAngles(azimuth: azimuthStep * Double(step.count), elevation: 0))
        }
        return ModelOutput(step: step, tokens: tokens, predictedTokens: predicted, command: command, wellness: Float(step.count))
    }

    func reset() {
        filledContextSteps = 0
        previousExecuted = []
    }
}

/// A planner whose goal ring count is set by the test and, with `fillsRing`, grows by one
/// entry per record up to 200; its plan sets wrist_L y = 0.5 and scores 21 evaluations
/// 21, 20, ..., 1, selecting the last; it returns `fingerGoals` as the hand descent's goals.
final class StubPlanner: Planning {
    var goalRingCount: Int
    let fillsRing: Bool
    var fingerGoals: FingerJointGoals? = nil
    var records = 0
    var planSteps: [ModelStep] = []
    var resets = 0

    init(goalRingCount: Int, fillsRing: Bool = false) {
        self.goalRingCount = goalRingCount
        self.fillsRing = fillsRing
    }

    func record(tokens: TokenSet, wellness: Float, at step: ModelStep) {
        records += 1
        if fillsRing {
            goalRingCount = min(goalRingCount + 1, ModelStep.goalRingLength)
        }
    }

    func plan(from command: Command, at step: ModelStep) throws -> PlanResult {
        planSteps.append(step)
        var planned = command
        planned[.wristL, .positionY] = 0.5
        let scores = (0 ..< PlanResult.evaluationsPerEvent).map { Float(PlanResult.evaluationsPerEvent - $0) }
        return try PlanResult(
            command: planned, scores: scores, selectedIteration: PlanResult.evaluationsPerEvent - 1,
            goalStep: ModelStep(count: 0), goalWellness: 1, fingerGoals: fingerGoals)
    }

    func reset() {
        resets += 1
    }
}

/// A decoder that records each command, gaze pair and finger goal set it reads and returns
/// 92 zeros.
final class StubDecoder: CommandDecoding {
    let motorContract = "ancestor_arm_axes_signed_limits_v3+finger_joint_servo_v1"
    var decoded: [Command] = []
    var gazes: [GazeAngles?] = []
    var fingerGoalsSeen: [FingerJointGoals?] = []

    func decode(_ command: Command, mask: HandleMask, gaze: GazeAngles?, fingerGoals: FingerJointGoals?) throws -> MotorControls {
        guard command.scale == .centimetres else {
            throw UEMACoreError.wrongScale(expected: .centimetres, found: command.scale)
        }
        decoded.append(command)
        gazes.append(gaze)
        fingerGoalsSeen.append(fingerGoals)
        return MotorControls.zero
    }
}

/// An execution transform that copies the 234 numbers and relabels them as centimetres;
/// with `writesStraightAheadGaze` it also sets the eyes' row to azimuth 0, elevation 0. It
/// keeps the protocol's default `execute`, so the planner's finger goals pass through.
final class StubExecutor: CommandExecuting {
    var calls = 0
    var writesStraightAheadGaze = false

    func executedCommand(fromRaw raw: Command, achievedMotor: SenseReading, at step: ModelStep) throws -> Command {
        calls += 1
        var executed = try Command(values: raw.values, scale: .centimetres)
        if writesStraightAheadGaze {
            executed.setGaze(.straightAhead)
        }
        return executed
    }

    func reset() {}
}

/// An execution transform that makes finger goals on every model step, as the Python model
/// does under hand_target_mode wrist_relative_v1 (lem_model.py:4525-4534): 44 angles all
/// equal to the model step's count, in radians. It records the planner's goals it was given.
final class StubFingerGoalExecutor: CommandExecuting {
    var plannedGoalsSeen: [FingerJointGoals?] = []

    func executedCommand(fromRaw raw: Command, achievedMotor: SenseReading, at step: ModelStep) throws -> Command {
        try Command(values: raw.values, scale: .centimetres)
    }

    func execute(
        raw: Command, plannedFingerGoals: FingerJointGoals?, achievedMotor: SenseReading, at step: ModelStep
    ) throws -> ExecutionResult {
        plannedGoalsSeen.append(plannedFingerGoals)
        return ExecutionResult(
            command: try executedCommand(fromRaw: raw, achievedMotor: achievedMotor, at: step),
            fingerGoals: try FingerJointGoals(
                angles: Array(repeating: Double(step.count), count: FingerJointGoals.count)))
    }

    func reset() {}
}

/// A renderer returning a 1 x 1 black image per camera.
final class StubEyeRenderer: EyeRendering {
    var renders = 0

    func render(camera: EyeCamera, pose: CameraPose, geoms: [SceneGeom]) throws -> EyeImage {
        renders += 1
        return try EyeImage(camera: camera, width: 1, height: 1, rgb: [0, 0, 0])
    }
}

/// No geoms; every camera at the origin with the identity rotation.
final class StubSceneGeoms: SceneGeomReading {
    func sceneGeoms() throws -> [SceneGeom] {
        []
    }

    func cameraPose(_ camera: EyeCamera) throws -> CameraPose {
        try CameraPose(position: [0, 0, 0], rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1])
    }
}

/// Memory and thermal readings set by the test.
final class StubMemoryStatus: MemoryStatusReading {
    var availableMemoryBytes: Int?
    var thermalState: ProcessInfo.ThermalState

    init(availableMemoryBytes: Int?, thermalState: ProcessInfo.ThermalState) {
        self.availableMemoryBytes = availableMemoryBytes
        self.thermalState = thermalState
    }
}

/// Keeps every saved snapshot.
final class StubStateSaver: SessionStateSaving {
    var saved: [SessionSnapshot] = []

    func save(_ snapshot: SessionSnapshot) throws {
        saved.append(snapshot)
    }
}

/// One session over the fakes, with the four observing view models.
@MainActor
final class SessionHarness {
    let physics: StubPhysics
    let readers: [StubSenseReader]
    let model: StubForwardModel
    let planner: StubPlanner
    let decoder: StubDecoder
    let executor: StubExecutor
    let renderer: StubEyeRenderer
    let geoms: StubSceneGeoms
    let memoryStatus: StubMemoryStatus
    let saver: StubStateSaver
    let memory: MemoryViewModel
    let senses: SensesViewModel
    let commands: CommandViewModel
    let plans: PlannerViewModel
    let pipeline: ModelStepPipeline
    let run: RunViewModel

    init(
        thermalState: ProcessInfo.ThermalState = .nominal, goalRingCount: Int = ModelStep.goalRingLength,
        fillsGoalRing: Bool = false, eyeImageInterval: Int = SessionCadence.releasedEyeImageInterval,
        executorOverride: (any CommandExecuting)? = nil
    ) throws {
        // Every part is made as a local first: a class init cannot read its own stored
        // properties before all of them are set.
        let physics = StubPhysics()
        let readers = SenseName.active.map { StubSenseReader(sense: $0) }
        let model = StubForwardModel()
        let planner = StubPlanner(goalRingCount: goalRingCount, fillsRing: fillsGoalRing)
        let decoder = StubDecoder()
        let executor = StubExecutor()
        let renderer = StubEyeRenderer()
        let geoms = StubSceneGeoms()
        let memoryStatus = StubMemoryStatus(availableMemoryBytes: nil, thermalState: thermalState)
        let saver = StubStateSaver()
        let memory = MemoryViewModel(monitor: memoryStatus)
        let senses = SensesViewModel()
        let commands = CommandViewModel()
        let plans = PlannerViewModel()
        let pipeline = try ModelStepPipeline(
            physics: physics,
            readers: readers,
            tokenizers: SenseName.active.map { StubTokenizer(sense: $0) },
            model: model,
            planner: planner,
            decoder: decoder,
            executor: executorOverride ?? executor,
            eyeRenderer: renderer,
            sceneGeoms: geoms,
            planningPermit: memory,
            eyeImageInterval: eyeImageInterval)
        self.physics = physics
        self.readers = readers
        self.model = model
        self.planner = planner
        self.decoder = decoder
        self.executor = executor
        self.renderer = renderer
        self.geoms = geoms
        self.memoryStatus = memoryStatus
        self.saver = saver
        self.memory = memory
        self.senses = senses
        self.commands = commands
        self.plans = plans
        self.pipeline = pipeline
        self.run = RunViewModel(runner: pipeline, observers: [senses, commands, plans, memory], stateSaver: saver)
    }

    func runModelSteps(_ count: Int) -> [ModelStepReport] {
        var reports: [ModelStepReport] = []
        for _ in 0 ..< count {
            guard run.stepOnce(), let report = run.lastReport else { break }
            reports.append(report)
        }
        return reports
    }
}

// MARK: - Tests

final class RunViewModelTests: XCTestCase {
    /// Acceptance: with a goal ring of 200 entries, model steps 0 to 9 hold one planning
    /// candidate (count 0), so the planner runs once and its 21 scores are published.
    @MainActor
    func testTenModelStepsCallThePlannerOnce() throws {
        let harness = try SessionHarness()
        let reports = harness.runModelSteps(10)

        XCTAssertNil(harness.run.lastErrorDescription)
        XCTAssertEqual(reports.count, 10)
        XCTAssertEqual(harness.run.completedModelSteps, 10)
        XCTAssertEqual(harness.planner.planSteps, [ModelStep(count: 0)])
        XCTAssertEqual(harness.planner.records, 10)
        XCTAssertEqual(harness.run.plannerScores.count, PlanResult.evaluationsPerEvent)
        XCTAssertEqual(harness.plans.eventsRun, 1)
        XCTAssertEqual(harness.plans.selectedIteration, PlanResult.evaluationsPerEvent - 1)
        XCTAssertEqual(reports.filter { $0.planningEvent != .notDue }.count, 1)
    }

    /// Acceptance: at thermal state critical the due event of model step 0 is skipped and
    /// the motor controller's command is executed instead.
    @MainActor
    func testCriticalThermalStateSkipsThePlanningEvent() throws {
        let harness = try SessionHarness(thermalState: .critical)
        let reports = harness.runModelSteps(10)

        XCTAssertEqual(reports.count, 10)
        XCTAssertTrue(harness.planner.planSteps.isEmpty)
        XCTAssertEqual(reports.first?.planningEvent, .skipped)
        XCTAssertEqual(harness.memory.planningEventsSkipped, 1)
        XCTAssertEqual(harness.plans.eventsSkipped, 1)
        XCTAssertEqual(harness.pipeline.planningEventsSkipped, 1)
        XCTAssertTrue(harness.run.plannerScores.isEmpty)
        XCTAssertEqual(reports.first?.rawCommand, reports.first?.output.command)
    }

    /// Serious and critical block the event; nominal and fair permit it.
    func testSeriousAndCriticalBlockNominalAndFairPermit() {
        XCTAssertFalse(MemoryViewModel.blocksPlanning(.nominal))
        XCTAssertFalse(MemoryViewModel.blocksPlanning(.fair))
        XCTAssertTrue(MemoryViewModel.blocksPlanning(.serious))
        XCTAssertTrue(MemoryViewModel.blocksPlanning(.critical))
        XCTAssertEqual(MemoryViewModel.name(of: .serious), "serious")
    }

    /// Acceptance: a background notice stops the running loop, no model step runs after
    /// it, and the snapshot taken at the notice is saved once.
    @MainActor
    func testBackgroundNoticeStopsTheLoop() async throws {
        let harness = try SessionHarness()
        harness.run.start()
        XCTAssertTrue(harness.run.isRunning)
        // Yield until the loop has completed at least one model step (at most 1,000 yields).
        var yields = 0
        while harness.run.completedModelSteps == 0, yields < 1_000 {
            await Task.yield()
            yields += 1
        }
        XCTAssertGreaterThan(harness.run.completedModelSteps, 0)

        harness.run.handleScenePhase(.background)
        let countAtNotice = harness.run.completedModelSteps
        XCTAssertFalse(harness.run.isRunning)
        for _ in 0 ..< 20 {
            await Task.yield()
        }

        XCTAssertEqual(harness.run.completedModelSteps, countAtNotice)
        XCTAssertEqual(harness.saver.saved.count, 1)
        XCTAssertEqual(harness.saver.saved.first?.completedModelSteps, countAtNotice)
        XCTAssertEqual(harness.run.lastSavedSnapshot?.completedModelSteps, countAtNotice)
    }

    /// One model step: 6 reads, 5 decodes, 5 control writes and 10 physics steps, each
    /// reader told of every physics step and every env step.
    @MainActor
    func testOneModelStepRunsTenPhysicsStepsAndFiveDecodes() throws {
        let harness = try SessionHarness()
        let reports = harness.runModelSteps(1)

        XCTAssertEqual(reports.first?.controls.count, ModelStep.envStepsPerModelStep)
        XCTAssertEqual(harness.physics.physicsSteps, ModelStep.physicsStepsPerModelStep)
        XCTAssertEqual(harness.physics.writtenControls.count, ModelStep.envStepsPerModelStep)
        XCTAssertEqual(harness.decoder.decoded.count, ModelStep.envStepsPerModelStep)
        for reader in harness.readers {
            XCTAssertEqual(reader.reads, 1)
            XCTAssertEqual(reader.physicsStepsSeen, ModelStep.physicsStepsPerModelStep)
            XCTAssertEqual(reader.envStepsSeen, ModelStep.envStepsPerModelStep)
        }
        XCTAssertEqual(harness.run.tokens.count, SenseName.activeCount)
    }

    /// With an empty goal ring that fills one entry per model step, model steps 0 to 30
    /// (31 model steps) plan at counts 20 and 30: the ring holds 20 entries from count 19
    /// on (the record precedes the test, planner_loss.md section 1), and 20 and 30 are the
    /// multiples of 10 from there (lem_model.py:4283).
    @MainActor
    func testFillingGoalRingPlansAtModelSteps20And30() throws {
        let harness = try SessionHarness(goalRingCount: 0, fillsGoalRing: true)
        let reports = harness.runModelSteps(31)

        XCTAssertEqual(reports.count, 31)
        XCTAssertEqual(harness.planner.planSteps, [ModelStep(count: 20), ModelStep(count: 30)])
        XCTAssertEqual(reports.last?.goalRingCount, 31)
    }

    /// With 19 entries in the goal ring no planning event is due.
    @MainActor
    func testNoPlanningEventBeforeTheGoalRingHoldsTwenty() throws {
        let harness = try SessionHarness(goalRingCount: ModelStep.goalRingMinimumForPlanning - 1)
        let reports = harness.runModelSteps(10)

        XCTAssertEqual(reports.count, 10)
        XCTAssertTrue(harness.planner.planSteps.isEmpty)
        XCTAssertTrue(reports.allSatisfy { $0.planningEvent == .notDue })
        XCTAssertEqual(harness.memory.planningEventsSkipped, 0)
    }

    /// The executed command is latched at model steps 0 and 5 and held in between; the
    /// model receives no previous command at model step 0 and the held one after.
    @MainActor
    func testCommandIsHeldForFiveModelSteps() throws {
        let harness = try SessionHarness()
        let reports = harness.runModelSteps(10)

        XCTAssertEqual(reports.map { $0.latchedNewCommand },
                       [true, false, false, false, false, true, false, false, false, false])
        XCTAssertEqual(reports[3].sentCommand, reports[0].executedCommand)
        XCTAssertEqual(reports[5].sentCommand, reports[5].executedCommand)
        XCTAssertNotEqual(reports[5].sentCommand, reports[0].sentCommand)
        XCTAssertEqual(reports[0].sentCommand[.wristL, .positionY], 0.5)
        XCTAssertNil(harness.model.previousExecuted.first ?? nil)
        // The motor slot of model step 1: model step 0's held command with the burst's gaze
        // (from the raw command) in its eyes' row (train_mimo.py:3292-3294).
        var expectedPrevious = reports[0].sentCommand
        expectedPrevious.setGaze(reports[0].rawCommand.gazeAngles)
        XCTAssertEqual(harness.model.previousExecuted[1], expectedPrevious)
        XCTAssertEqual(harness.commands.sentRows.count, BodyHandle.count)
        XCTAssertEqual(harness.commands.sentRows.first?.values.count, Command.numbersPerHandle)
    }

    /// The decoder's gaze pair comes from each model step's RAW command and is not held
    /// (lem_model.py:3128; train_mimo.py:3040, :3244): the executor here sets the executed
    /// eyes' row straight ahead, and model step 1 holds model step 0's command, yet the
    /// decoder reads azimuth 0.1 rad at model step 1 and 0.2 rad at model step 2. The motor
    /// slot of model step 2 is model step 0's held command with model step 1's gaze.
    @MainActor
    func testDecoderGazeComesFromTheRawCommandAndIsNotHeld() throws {
        let harness = try SessionHarness(goalRingCount: 0)
        harness.model.gazeAzimuthPerModelStep = 0.1
        harness.executor.writesStraightAheadGaze = true
        let reports = harness.runModelSteps(3)

        XCTAssertEqual(reports.count, 3)
        let gazes = harness.decoder.gazes
        XCTAssertEqual(gazes.count, 3 * ModelStep.envStepsPerModelStep)
        XCTAssertFalse(reports[1].latchedNewCommand)
        XCTAssertEqual(reports[1].executedCommand.gazeAngles.azimuth, 0, accuracy: 1e-9)
        XCTAssertEqual(gazes[5]?.azimuth ?? .nan, 0.1, accuracy: 1e-5)
        XCTAssertEqual(gazes[9]?.azimuth ?? .nan, 0.1, accuracy: 1e-5)
        XCTAssertEqual(gazes[10]?.azimuth ?? .nan, 0.2, accuracy: 1e-5)

        let previous = try XCTUnwrap(harness.model.previousExecuted[2])
        XCTAssertEqual(previous.gazeAngles.azimuth, 0.1, accuracy: 1e-5)
        XCTAssertEqual(previous[.wristL, .positionX], reports[0].sentCommand[.wristL, .positionX])
    }

    /// The finger joint goals the execution step gives are latched with the command: with
    /// goals made every model step (44 angles equal to the count), decodes of model steps 0
    /// to 4 carry model step 0's goals and those of model steps 5 and 6 carry model step 5's.
    /// With the default `execute`, the planner's goals of model step 0 are held to model step
    /// 4 and model step 5 latches none.
    @MainActor
    func testExecutionFingerGoalsAreLatchedWithTheCommand() throws {
        let fingerExecutor = StubFingerGoalExecutor()
        let everyStep = try SessionHarness(executorOverride: fingerExecutor)
        _ = everyStep.runModelSteps(7)
        XCTAssertEqual(fingerExecutor.plannedGoalsSeen.count, 7)
        let seen = everyStep.decoder.fingerGoalsSeen
        XCTAssertEqual(seen.count, 7 * ModelStep.envStepsPerModelStep)
        XCTAssertEqual(seen[0]?.angles.first, 0)
        XCTAssertEqual(seen[24]?.angles.first, 0)
        XCTAssertEqual(seen[25]?.angles.first, 5)
        XCTAssertEqual(seen[34]?.angles.first, 5)
        XCTAssertEqual(everyStep.pipeline.snapshot().heldFingerGoals?.angles.first, 5)

        let planned = try FingerJointGoals(angles: Array(repeating: 0.25, count: FingerJointGoals.count))
        let passThrough = try SessionHarness()
        passThrough.planner.fingerGoals = planned
        _ = passThrough.runModelSteps(6)
        let passed = passThrough.decoder.fingerGoalsSeen
        XCTAssertEqual(passed.count, 6 * ModelStep.envStepsPerModelStep)
        XCTAssertEqual(passed[0], planned)
        XCTAssertEqual(passed[24], planned)
        XCTAssertNil(passed[25])
    }

    /// Five readers, a duplicated sense or an intermind reader are refused at init.
    @MainActor
    func testWrongReaderSetIsRefused() throws {
        let tokenizers: [any Tokenizer] = SenseName.active.map { StubTokenizer(sense: $0) }
        func pipeline(readers: [any SenseReader]) throws -> ModelStepPipeline {
            try ModelStepPipeline(
                physics: StubPhysics(), readers: readers, tokenizers: tokenizers, model: StubForwardModel(),
                planner: StubPlanner(goalRingCount: 0), decoder: StubDecoder(), executor: StubExecutor(),
                eyeRenderer: StubEyeRenderer(), sceneGeoms: StubSceneGeoms(),
                planningPermit: MemoryViewModel(monitor: StubMemoryStatus(availableMemoryBytes: nil, thermalState: .nominal)))
        }
        let five: [any SenseReader] = SenseName.active.dropLast().map { StubSenseReader(sense: $0) }
        XCTAssertThrowsError(try pipeline(readers: five)) { error in
            XCTAssertEqual(error as? SessionConfigurationError,
                           .wrongPartCount(part: "SenseReader", expected: 6, found: 5))
        }
        let duplicated: [any SenseReader] = SenseName.active.map { _ in StubSenseReader(sense: .vision) }
        XCTAssertThrowsError(try pipeline(readers: duplicated)) { error in
            XCTAssertEqual(error as? SessionConfigurationError, .duplicateSense(part: "SenseReader", sense: .vision))
        }
        let withIntermind: [any SenseReader] = five + [StubSenseReader(sense: .intermind)]
        XCTAssertThrowsError(try pipeline(readers: withIntermind)) { error in
            XCTAssertEqual(error as? SessionConfigurationError, .inactiveSense(part: "SenseReader", sense: .intermind))
        }
    }

    /// The prediction error is nil at model step 0 and, at model step 1, the mean squared
    /// difference between the predicted tokens of step 0 (all 1) and the tokens of step 1
    /// (all 1): 0. The heat map is 12 rows of 16; the input range is -1 to the count.
    @MainActor
    func testSensesViewModelPredictionErrorAndHeatMap() throws {
        let harness = try SessionHarness()
        _ = harness.runModelSteps(1)
        XCTAssertEqual(harness.senses.rows.count, SenseName.activeCount)
        XCTAssertTrue(harness.senses.rows.allSatisfy { $0.predictionError == nil })

        _ = harness.runModelSteps(1)
        XCTAssertTrue(harness.senses.rows.allSatisfy { $0.predictionError == 0 })
        let first = try XCTUnwrap(harness.senses.rows.first)
        XCTAssertEqual(first.sense, .vision)
        XCTAssertEqual(first.heatMap.count, SensePanelRow.heatMapRows)
        XCTAssertTrue(first.heatMap.allSatisfy { $0.count == SensePanelRow.heatMapColumns })
        XCTAssertEqual(first.inputShape, [2])
        XCTAssertEqual(first.inputMinimum, -1)
        XCTAssertEqual(first.inputMaximum, 1)
    }

    /// 3 MB at launch, then 2 MB: 2.0 MB available and a 1.0 MB peak footprint; a rise to
    /// 2.5 MB leaves the peak at 1.0 MB.
    @MainActor
    func testMemoryViewModelReportsMegabytesAndPeakFootprint() {
        let status = StubMemoryStatus(availableMemoryBytes: 3 * 1_048_576, thermalState: .fair)
        let memory = MemoryViewModel(monitor: status)
        XCTAssertEqual(memory.launchAvailableMemoryMB, 3.0)
        XCTAssertEqual(memory.peakFootprintMB, 0)

        status.availableMemoryBytes = 2 * 1_048_576
        memory.refresh()
        XCTAssertEqual(memory.availableMemoryMB, 2.0)
        XCTAssertEqual(memory.peakFootprintMB, 1.0)

        status.availableMemoryBytes = 5 * 524_288
        memory.refresh()
        XCTAssertEqual(memory.availableMemoryMB, 2.5)
        XCTAssertEqual(memory.peakFootprintMB, 1.0)
        XCTAssertTrue(memory.permitsPlanningEvent())
        XCTAssertEqual(memory.thermalStateName, "fair")
    }

    /// At interval 1 the five cameras render every model step; at interval 2 on model
    /// step 0 only of steps 0 and 1.
    @MainActor
    func testEyeImagesFollowTheirInterval() throws {
        let everyStep = try SessionHarness()
        _ = everyStep.runModelSteps(2)
        XCTAssertEqual(everyStep.renderer.renders, 2 * EyeCamera.allCases.count)
        XCTAssertEqual(everyStep.run.eyeImages.count, EyeCamera.allCases.count)

        let everySecond = try SessionHarness(eyeImageInterval: 2)
        let reports = everySecond.runModelSteps(2)
        XCTAssertEqual(everySecond.renderer.renders, EyeCamera.allCases.count)
        XCTAssertTrue(reports[1].eyeImages.isEmpty)
        XCTAssertEqual(everySecond.run.eyeImages.count, EyeCamera.allCases.count)
    }

    /// A snapshot written by JSONSessionStateStore reads back equal.
    @MainActor
    func testSnapshotRoundTripsThroughJSONStore() throws {
        let harness = try SessionHarness()
        _ = harness.runModelSteps(3)
        let snapshot = harness.pipeline.snapshot()
        XCTAssertEqual(snapshot.completedModelSteps, 3)
        XCTAssertEqual(snapshot.planningEventsRun, 1)

        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("wi32_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let store = JSONSessionStateStore(fileURL: folder.appendingPathComponent("session_snapshot.json"))
        XCTAssertNil(try store.load())
        try store.save(snapshot)
        XCTAssertEqual(try store.load(), snapshot)
    }
}
