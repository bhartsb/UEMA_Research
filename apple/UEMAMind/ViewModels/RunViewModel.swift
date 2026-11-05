// RunViewModel.swift: the session loop of UEMA Mind. One model step (one 50 ms model
// update) runs the stages of plan section 2 in sequence over injected protocol types, and
// RunViewModel starts, pauses and single-steps that loop and publishes what each model
// step produced: the model-step count, the six tokens, the command, the planner's scores
// and the five eye images. Work item WI-32, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; this is the work item's main
// file. The other four view models (SensesViewModel, CommandViewModel, PlannerViewModel,
// MemoryViewModel) sit beside it in apple/UEMAMind/ViewModels/ and receive every model
// step's report through `ModelStepObserving`.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). Every UEMAKit dependency is a UEMACore protocol
// (apple/UEMAKit/Sources/UEMACore/Protocols.swift, WI-20), so the tests of
// apple/UEMAMind/Tests/ run the loop on fakes without MuJoCo, MLX or SceneKit.
//
// TYPES IN THIS FILE, one responsibility each:
//   CommandExecuting       protocol: the execution transform, raw command (.normalized)
//                          into the executed command (.centimetres)
//   PlanningPermitting     protocol: whether a due planning event may run now
//                          (MemoryViewModel implements it from the thermal state)
//   ModelStepObserving     protocol: receives each model step's report (the other view
//                          models implement it)
//   SessionStateSaving     protocol: writes the session snapshot on a background notice
//   ModelStepRunning       protocol: runs one model step and returns its report
//   PlanningEventOutcome   enum: what happened to the planning event of one model step
//   ModelStepReport        struct: everything one model step produced
//   SessionSnapshot        struct: the loop's own state, saved on a background notice
//   SessionCadence         enum: the released hold and eye image cadences
//   SessionConfigurationError, SessionStepError   the errors of this file
//   ModelStepPipeline      class: the stage sequence of one model step (ModelStepRunning)
//   JSONSessionStateStore  class: SessionStateSaving into one JSON file
//   RunViewModel           class: loop control and publication (ObservableObject)
//
// THE SEQUENCE OF ONE MODEL STEP (ModelStepPipeline.runModelStep). The work item lists
// "per model step physics 10 steps, read, tokenize, model, command (planner every 10th),
// decode, write". The loop is a cycle; this file starts each model step at "read" so that
// the first model step reads the state the physics was reset to, as the Python trainer
// does: train_mimo.py:2923 calls `model.step(frames, ...)` on the frames of the last
// burst, and the burst of `action_repeat` env steps follows at train_mimo.py:3243-3250.
//   1. read: each of the six SenseReaders returns its SenseReading for model step `step`.
//   2. tokenize: each sense's Tokenizer turns the reading into its EncoderInput.
//   3. model: ForwardModeling.step encodes the six inputs, rewrites the previous model
//      step's motor slot with the command sent in the last burst, its eyes' row holding the
//      burst's gaze (stage 8; nil on the first model step; Protocols.swift
//      ForwardModeling.step doc, proprio_rows.md section 5 step 3),
//      and returns the six tokens, the six predicted tokens, the motor controller's
//      command (.normalized) and the wellness channel sum.
//   4. goal ring: Planning.record(tokens:wellness:at:) appends the model step
//      (lem_model.py:4271-4272 per planner_loss.md section 1; ring of 200,
//      lem_model.py:998).
//   5. command: when Planning.canPlan(at:) is true (ring of 20 or more and
//      step.count % 10 == 0, lem_model.py:4283) AND PlanningPermitting allows it, the
//      planner's command replaces the motor controller's; when the permit refuses, the
//      event is skipped and the motor controller's command is used (work item: "at
//      serious or critical the planner's event is skipped").
//   6. execution: CommandExecuting.execute turns the raw command into the executed command
//      in centimetres (planner_loss.md section 3 step 2: "The result is in execution units
//      (cm)") and gives this model step's 44 finger joint goals. The Python model makes
//      those goals on EVERY model step under `hand_target_mode` wrist_relative_v1
//      (args.json): the planner's hand descent goals on a planning event, otherwise
//      `hand_kinematics.project_world_orders` of the executed command, then
//      `_smooth_hand_joints` at `smooth_actions` 0.35 (lem_model.py:4525-4534, 6046-6050),
//      and it writes them back into the finger rows (`assemble_world`, lem_model.py:4560-
//      4561). The default `execute` passes the planner's goals through (nil on other model
//      steps); the concrete execution transform replaces it with the Python sequence.
//   7. hold: one executed command is latched when step.count % 5 == 0 (or none is held
//      yet) and sent for 5 model steps (`motor_hold` 5, weights/
//      b1_continue_B1K_skip13_492800/run/args.json:31; train_mimo.py:3038-3044: "One
//      ORDER, held for motor_hold model steps"; "if step % hold == 0 or held_motor is
//      None:"). The finger joint goals are latched with it (`held_hand_joint_goal =
//      res.hand_joint_goal`, train_mimo.py:3047). Gaze is not held (train_mimo.py:3040:
//      "Gaze is NOT held"): the burst passes `res.gaze_out` (train_mimo.py:3244), which is
//      the gaze read from the RAW command's eyes' row, the motor controller's
//      (`gaze_out = self._gaze_from_handle(motor_out)`, lem_model.py:3128) or the plan's
//      (lem_model.py:4401), before the execution transform; `goal_gaze_hold` is off
//      (args.json), so lem_model.py:3129-3134 does not replace it.
//   8. decode and write, 5 times: CommandDecoding.decode makes the 92 controls of one env
//      step from the held command and the body's current state, and
//      Physics.stepEnvStep(controls:afterEachPhysicsStep:) writes them and runs 2 physics
//      steps, calling every reader's observePhysicsStep after each; every reader's
//      observeEnvStep runs after each env step. 5 env steps x 2 physics steps = 10
//      physics steps (ModelStep.physicsStepsPerModelStep; command_decoder.md section 1:
//      "The decoder runs once per call of `BridgeEnv.step` ..., which the trainer calls 5
//      times per model step"). After the burst, the command the next model step's motor
//      slot receives is the held command with its eyes' row rewritten to the gaze of the
//      burst (`prev_executed = LemLite.gaze_to_handle(send_motor, gaze_seq.mean(axis=0),
//      ...)`, train_mimo.py:3292-3294; Command.setGaze is gaze_to_handle's 'legacy'
//      convention). The Python takes the mean of the eyes' realised angles over the burst;
//      this loop has only the commanded pair, which it uses.
//   9. eye images: on a model step whose count is a multiple of `eyeImageInterval`
//      (default 1), the five cameras are rendered from SceneGeomReading's geoms and
//      poses through EyeRendering, after the burst: train_mimo.py:3244-3245 renders on
//      the last env step of every burst (`render=(rep == reps - 1)`), the image the next
//      model step reads. The plan's section 2 says "every 2nd model step"; the trainer's
//      rule is the default here and the interval is an init parameter.
//
// MODEL STEP NUMBERING. The model step being run has the count of model steps completed
// before it: the first is ModelStep(count: 0), as the trainer's counter starts at 0
// (train_mimo.py:1806, "step = 0"), is passed as `step_index=step` (train_mimo.py:2923)
// and is incremented after the burst (train_mimo.py:3378, "step += 1"). With a goal ring
// holding 20 or more entries, 10 model steps (counts 0 to 9) therefore contain exactly one
// planning candidate, count 0.
//
// THE BACKGROUND NOTICE. On `ScenePhase.background` the loop stops and the
// SessionSnapshot is written synchronously inside the call: "That method has five seconds
// to perform any tasks and return. Shortly after that method returns, the system puts
// your app into the suspended state." (docs/apple_port/evidence/research1.json, quoting
// developer.apple.com/documentation/uikit/extending-your-app-s-background-execution-time).
// The snapshot holds the loop's own state (count, held command, finger goals, last
// controls, event counts). The forward model's 128-model-step context and the goal ring's
// 200 entries are not in it: ForwardModeling and Planning declare no way to read them out.
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave:
// - work_items.json entry WI-32 and plan row WI-32: "RunViewModel owns the loop, injected
//   with Physics, six SenseReaders, Tokenizers, ForwardModeling, Planning,
//   CommandDecoding, EyeRendering (protocols only): per model step physics 10 steps, read,
//   tokenize, model, command (planner every 10th), decode, write; it publishes the
//   model-step count, tokens, command, planner scores, eye images." Acceptance: "with
//   fakes, 10 model steps call the planner once; a fake critical thermal state skips it;
//   a background notice stops the loop."
// - apple/UEMAKit/Sources/UEMACore/Protocols.swift (WI-20): Physics, SenseReader,
//   Tokenizer, ForwardModeling, Planning (canPlan default), CommandDecoding, EyeRendering,
//   SceneGeomReading, ModelOutput, PlanResult; Physics.stepEnvStep's doc: "the session
//   calls this 5 times per model step with a fresh decode".
// - apple/UEMAKit/Sources/UEMACore/Command.swift: CommandScale.centimetres "reached from
//   `normalized` through the execution transform (planner_loss.md section 3 step 2),
//   which is the session's step, not a multiplication by 200"; HandleMask.allTrue "The
//   released run's mask: all 26 true".
// - apple/UEMAKit/Specs/planner_loss.md section 3 step 2 (the execution transform:
//   differentiable_execution_transform, lem_model.py:5664-5727, then smooth_blend 0.35,
//   lem_model.py:5142-5165) and section 6 point C: "lem_model.py:6429 onward ... and
//   :4336-4481 (executing the plan, babble, the ledger) are the session loop's
//   responsibility".
// - Source/Python/mimo_bridge/train_mimo.py lines 1806, 2923, 3030-3047, 3243-3250, 3378
//   (quoted above).
// - docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html section 5: "StepScheduler;
//   SessionRecorder, MemoryMonitor | Cadences (render 2, plan 10, tactile every substep)"
//   and REQ-B-14 (the screens this file feeds).
//
// FIRST BUILD ON A MAC (this work item's acceptance: "with fakes, 10 model steps call the
// planner once; a fake critical thermal state skips it; a background notice stops the
// loop"):
//  1. Run WI-17's and WI-33's steps first: the MuJoCo XCFramework in place, `swift package
//     resolve` exit 0 in apple/UEMAKit, `cd apple && xcodegen generate` exit 0. The app
//     targets compile every file under apple/UEMAMind except Tests/ (apple/project.yml), so
//     WI-38's UEMAMindApp.swift and Views/ must be on disk for the app to link.
//  2. `xcodebuild -scheme UEMAMind-macOS -destination 'platform=macOS' build` : exit 0. A
//     compile error naming a UEMACore requirement means this file is out of date with
//     Protocols.swift, and Protocols.swift wins.
//  3. `xcodebuild -scheme UEMAMind-macOS -destination 'platform=macOS' test` : exit 0 with
//     every test of apple/UEMAMind/Tests/RunViewModelTests.swift passing (15 tests; the
//     three acceptance tests are testTenModelStepsCallThePlannerOnce,
//     testCriticalThermalStateSkipsThePlanningEvent and testBackgroundNoticeStopsTheLoop).
//     The tests import the app module as `UEMAMind_macOS` (PRODUCT_NAME UEMAMind-macOS
//     with "-" made "_"); a "no such module" error means the module name differs: read
//     PRODUCT_MODULE_NAME in the generated project and correct the import.
//  4. Constructs to check first if step 2 fails, each used on purpose: protocols marked
//     `@MainActor` and classes conforming to them; `any Physics` passed where
//     `any PhysicsReading` is expected; a non-escaping closure passed to the protocol
//     extension `Physics.stepEnvStep`; `os_proc_available_memory()` inside `#if os(iOS)`
//     (MemoryViewModel.swift); `ScenePhase` compared with `==`; a `Task` started from a
//     main-actor method that captures `[weak self]`.
//  5. Record each step's command, exit code and clock time (AM/PM Pacific) in
//     apple/FIRST_BUILD_RECORD.md (WI-41).

import Combine
import Foundation
import SwiftUI
import UEMACore

// MARK: - Protocols the loop depends on

/// The execution transform (planner_loss.md section 3 step 2; lem_model.py:5664-5727 and
/// :5142-5165): the raw command of a model step (the motor controller's or the planner's,
/// scale `.normalized`) and the motor reading of the same model step (the achieved handle
/// state) into the executed command at scale `.centimetres`, which the decoder reads.
/// Class-bound: the transform keeps the previous proposal for its smoothing blend.
public protocol CommandExecuting: AnyObject {
    /// The executed command for `raw` at `step`; the result must be at `.centimetres`.
    func executedCommand(fromRaw raw: Command, achievedMotor: SenseReading, at step: ModelStep) throws -> Command

    /// The executed command and this model step's 44 finger joint goals (file header,
    /// stage 6; lem_model.py:4525-4534). `plannedFingerGoals` is the hand descent's output
    /// on a planning event that ran, nil otherwise. The default implementation below calls
    /// `executedCommand(fromRaw:achievedMotor:at:)` and passes `plannedFingerGoals` through.
    func execute(
        raw: Command, plannedFingerGoals: FingerJointGoals?, achievedMotor: SenseReading, at step: ModelStep
    ) throws -> ExecutionResult

    /// Clears the kept previous proposal.
    func reset()
}

extension CommandExecuting {
    public func execute(
        raw: Command, plannedFingerGoals: FingerJointGoals?, achievedMotor: SenseReading, at step: ModelStep
    ) throws -> ExecutionResult {
        ExecutionResult(
            command: try executedCommand(fromRaw: raw, achievedMotor: achievedMotor, at: step),
            fingerGoals: plannedFingerGoals)
    }
}

/// What the execution step made of one model step's raw command: the executed command
/// (scale `.centimetres`) and the 44 finger joint goals the decoder takes with it (nil
/// when the finger rows are decoded from their quaternions).
public struct ExecutionResult: Equatable {
    /// The executed command, positions in cm.
    public let command: Command

    /// The finger joint goals of this model step; nil when none.
    public let fingerGoals: FingerJointGoals?

    /// A result of the given values.
    public init(command: Command, fingerGoals: FingerJointGoals?) {
        self.command = command
        self.fingerGoals = fingerGoals
    }
}

/// Whether a planning event that is due may run now. MemoryViewModel implements it: false
/// at thermal state serious or critical.
@MainActor
public protocol PlanningPermitting: AnyObject {
    /// True when the due planning event runs; false when it is skipped.
    func permitsPlanningEvent() -> Bool
}

/// Receives the report of every model step the loop completes, in order.
@MainActor
public protocol ModelStepObserving: AnyObject {
    /// Called once per completed model step.
    func observe(_ report: ModelStepReport)

    /// Called when the session is reset to model step 0.
    func reset()
}

/// Writes the session snapshot (called on a background notice, inside the five seconds
/// before suspension).
public protocol SessionStateSaving: AnyObject {
    /// Writes `snapshot`; throws when it cannot be written.
    func save(_ snapshot: SessionSnapshot) throws
}

/// Runs the session one model step at a time.
@MainActor
public protocol ModelStepRunning: AnyObject {
    /// Model steps completed since the session started or was reset.
    var completedModelSteps: Int { get }

    /// Runs the next model step and returns what it produced.
    func runModelStep() throws -> ModelStepReport

    /// The loop's own state now.
    func snapshot() -> SessionSnapshot

    /// Resets the physics, the readers, the model, the planner and the loop state to
    /// model step 0.
    func reset() throws
}

// MARK: - Values one model step produces

/// What happened to the planning event of one model step.
public enum PlanningEventOutcome: Equatable {
    /// `Planning.canPlan(at:)` was false: not a multiple of 10, or the goal ring held
    /// fewer than 20 entries.
    case notDue
    /// The event ran; its result.
    case ran(PlanResult)
    /// The event was due and `PlanningPermitting` refused it; the motor controller's
    /// command was used.
    case skipped
}

/// Everything one model step produced, in the sequence the stages made it.
public struct ModelStepReport: Equatable {
    /// The model step (the first one has count 0).
    public let step: ModelStep

    /// The six readings, keyed by sense.
    public let readings: [SenseName: SenseReading]

    /// The six encoder inputs, keyed by sense.
    public let inputs: [SenseName: EncoderInput]

    /// The forward model's output: tokens, predicted tokens, the motor controller's
    /// command (.normalized) and the wellness channel sum.
    public let output: ModelOutput

    /// The planning event of this model step.
    public let planningEvent: PlanningEventOutcome

    /// The raw command the execution transform received (.normalized): the planner's when
    /// its event ran, else the motor controller's.
    public let rawCommand: Command

    /// This model step's executed command (.centimetres), before the hold.
    public let executedCommand: Command

    /// The command the decoder read for this model step's 5 env steps (.centimetres): the
    /// held one.
    public let sentCommand: Command

    /// True when `sentCommand` was latched at this model step.
    public let latchedNewCommand: Bool

    /// The 92 controls of each of the 5 env steps.
    public let controls: [MotorControls]

    /// The eye images rendered after the burst; empty on a model step without a render.
    public let eyeImages: [EyeCamera: EyeImage]

    /// The goal ring's entries after this model step's record, 0 to 200.
    public let goalRingCount: Int
}

/// The loop's own state, written on a background notice. The forward model's context and
/// the goal ring's entries are not part of it (see the file header).
public struct SessionSnapshot: Equatable, Codable {
    /// Model steps completed.
    public let completedModelSteps: Int

    /// The held executed command (.centimetres); nil before the first model step.
    public let heldCommand: Command?

    /// The finger joint goals latched with it; nil when none.
    public let heldFingerGoals: FingerJointGoals?

    /// The 92 controls of the last env step; nil before the first model step.
    public let lastControls: MotorControls?

    /// The goal ring's entries.
    public let goalRingCount: Int

    /// Planning events that ran.
    public let planningEventsRun: Int

    /// Planning events that were due and skipped.
    public let planningEventsSkipped: Int

    /// A snapshot of the given values.
    public init(
        completedModelSteps: Int, heldCommand: Command?, heldFingerGoals: FingerJointGoals?,
        lastControls: MotorControls?, goalRingCount: Int, planningEventsRun: Int, planningEventsSkipped: Int
    ) {
        self.completedModelSteps = completedModelSteps
        self.heldCommand = heldCommand
        self.heldFingerGoals = heldFingerGoals
        self.lastControls = lastControls
        self.goalRingCount = goalRingCount
        self.planningEventsRun = planningEventsRun
        self.planningEventsSkipped = planningEventsSkipped
    }
}

// MARK: - Errors

/// A set of injected parts that cannot make a session.
public enum SessionConfigurationError: Error, Equatable {
    /// The readers or tokenizers did not number 6 (one per active sense).
    case wrongPartCount(part: String, expected: Int, found: Int)
    /// Two readers or two tokenizers serve the same sense.
    case duplicateSense(part: String, sense: SenseName)
    /// A reader or tokenizer serves intermind, which the released run does not encode.
    case inactiveSense(part: String, sense: SenseName)
    /// A hold of 0 or less model steps, or an eye image interval below 0.
    case invalidCadence(name: String, value: Int)
}

/// A model step that could not be completed.
public enum SessionStepError: Error, Equatable {
    /// A reader returned a reading of another sense.
    case readingOfWrongSense(expected: SenseName, found: SenseName)
    /// No reading or no tokenizer for an active sense.
    case missingSense(SenseName)
}

// MARK: - Cadences

/// The released run's two session cadences that are init parameters of
/// ModelStepPipeline. A plain enum outside the main-actor classes, so that the constants
/// can be read from any context, default arguments included.
public enum SessionCadence {
    /// Model steps one executed command is held: 5 (`motor_hold` 5, args.json:31).
    public static let releasedMotorHoldModelSteps = 5

    /// Model steps between eye image renders: 1, the trainer's render on the last env
    /// step of every burst (train_mimo.py:3244-3245).
    public static let releasedEyeImageInterval = 1
}

// MARK: - One model step

/// The stage sequence of one model step over injected protocol types (the numbered
/// sequence in the file header). It holds the loop's own state: the completed count, the
/// held command and finger goals, the command sent in the last burst and the last
/// controls.
@MainActor
public final class ModelStepPipeline: ModelStepRunning {
    private let physics: any Physics
    private let readers: [any SenseReader]
    private let tokenizers: [SenseName: any Tokenizer]
    private let model: any ForwardModeling
    private let planner: any Planning
    private let decoder: any CommandDecoding
    private let executor: any CommandExecuting
    private let eyeRenderer: any EyeRendering
    private let sceneGeoms: any SceneGeomReading
    private let planningPermit: any PlanningPermitting
    private let handleMask: HandleMask
    private let motorHoldModelSteps: Int
    private let eyeImageInterval: Int

    /// Model steps completed since the session started or was reset.
    public private(set) var completedModelSteps = 0

    /// Planning events that ran.
    public private(set) var planningEventsRun = 0

    /// Planning events that were due and skipped.
    public private(set) var planningEventsSkipped = 0

    private var heldCommand: Command?
    private var heldFingerGoals: FingerJointGoals?
    private var lastSentCommand: Command?
    private var lastControls: MotorControls?

    /// A pipeline over the given parts. `readers` and `tokenizers` must each hold exactly
    /// one part per active sense (vision, tactile, audio, proprioception, motor,
    /// vestibular), in any sequence; readers run in the active sequence. `eyeImageInterval`
    /// 0 renders no eye images.
    public init(
        physics: any Physics,
        readers: [any SenseReader],
        tokenizers: [any Tokenizer],
        model: any ForwardModeling,
        planner: any Planning,
        decoder: any CommandDecoding,
        executor: any CommandExecuting,
        eyeRenderer: any EyeRendering,
        sceneGeoms: any SceneGeomReading,
        planningPermit: any PlanningPermitting,
        handleMask: HandleMask = .allTrue,
        motorHoldModelSteps: Int = SessionCadence.releasedMotorHoldModelSteps,
        eyeImageInterval: Int = SessionCadence.releasedEyeImageInterval
    ) throws {
        var readerBySense: [SenseName: any SenseReader] = [:]
        for reader in readers {
            guard reader.sense.isActive else {
                throw SessionConfigurationError.inactiveSense(part: "SenseReader", sense: reader.sense)
            }
            guard readerBySense[reader.sense] == nil else {
                throw SessionConfigurationError.duplicateSense(part: "SenseReader", sense: reader.sense)
            }
            readerBySense[reader.sense] = reader
        }
        guard readers.count == SenseName.activeCount else {
            throw SessionConfigurationError.wrongPartCount(
                part: "SenseReader", expected: SenseName.activeCount, found: readers.count)
        }
        var tokenizerBySense: [SenseName: any Tokenizer] = [:]
        for tokenizer in tokenizers {
            guard tokenizer.sense.isActive else {
                throw SessionConfigurationError.inactiveSense(part: "Tokenizer", sense: tokenizer.sense)
            }
            guard tokenizerBySense[tokenizer.sense] == nil else {
                throw SessionConfigurationError.duplicateSense(part: "Tokenizer", sense: tokenizer.sense)
            }
            tokenizerBySense[tokenizer.sense] = tokenizer
        }
        guard tokenizers.count == SenseName.activeCount else {
            throw SessionConfigurationError.wrongPartCount(
                part: "Tokenizer", expected: SenseName.activeCount, found: tokenizers.count)
        }
        guard motorHoldModelSteps >= 1 else {
            throw SessionConfigurationError.invalidCadence(name: "motorHoldModelSteps", value: motorHoldModelSteps)
        }
        guard eyeImageInterval >= 0 else {
            throw SessionConfigurationError.invalidCadence(name: "eyeImageInterval", value: eyeImageInterval)
        }
        self.physics = physics
        self.readers = SenseName.active.compactMap { readerBySense[$0] }
        self.tokenizers = tokenizerBySense
        self.model = model
        self.planner = planner
        self.decoder = decoder
        self.executor = executor
        self.eyeRenderer = eyeRenderer
        self.sceneGeoms = sceneGeoms
        self.planningPermit = planningPermit
        self.handleMask = handleMask
        self.motorHoldModelSteps = motorHoldModelSteps
        self.eyeImageInterval = eyeImageInterval
    }

    /// Runs the next model step: read, tokenize, model, goal ring, command, execution,
    /// hold, then 5 x (decode, write, 2 physics steps), then the eye images.
    public func runModelStep() throws -> ModelStepReport {
        let step = ModelStep(count: completedModelSteps)

        // 1. read
        var readings: [SenseName: SenseReading] = [:]
        for reader in readers {
            let reading = try reader.read(physics, at: step)
            guard reading.sense == reader.sense else {
                throw SessionStepError.readingOfWrongSense(expected: reader.sense, found: reading.sense)
            }
            readings[reader.sense] = reading
        }

        // 2. tokenize
        var inputs: [SenseName: EncoderInput] = [:]
        for sense in SenseName.active {
            guard let reading = readings[sense], let tokenizer = tokenizers[sense] else {
                throw SessionStepError.missingSense(sense)
            }
            inputs[sense] = try tokenizer.tokenize(reading)
        }

        // 3. model
        let output = try model.step(inputs: inputs, previousExecutedCommand: lastSentCommand, at: step)

        // 4. goal ring
        planner.record(tokens: output.tokens, wellness: output.wellness, at: step)

        // 5. command: the planner's on a due and permitted event
        var rawCommand = output.command
        var fingerGoalsNow: FingerJointGoals?
        let outcome: PlanningEventOutcome
        if planner.canPlan(at: step) {
            if planningPermit.permitsPlanningEvent() {
                let plan = try planner.plan(from: output.command, at: step)
                rawCommand = plan.command
                fingerGoalsNow = plan.fingerGoals
                planningEventsRun += 1
                outcome = .ran(plan)
            } else {
                planningEventsSkipped += 1
                outcome = .skipped
            }
        } else {
            outcome = .notDue
        }

        // 6. execution, with this model step's finger joint goals (lem_model.py:4525-4534)
        guard let motorReading = readings[.motor] else {
            throw SessionStepError.missingSense(.motor)
        }
        let execution = try executor.execute(
            raw: rawCommand, plannedFingerGoals: fingerGoalsNow, achievedMotor: motorReading, at: step)
        let executed = execution.command
        guard executed.scale == .centimetres else {
            throw UEMACoreError.wrongScale(expected: .centimetres, found: executed.scale)
        }

        // 7. hold (train_mimo.py:3044: "if step % hold == 0 or held_motor is None:")
        let sent: Command
        let latched: Bool
        if let held = heldCommand, step.count % motorHoldModelSteps != 0 {
            sent = held
            latched = false
        } else {
            heldCommand = executed
            heldFingerGoals = execution.fingerGoals
            sent = executed
            latched = true
        }
        // Gaze is not held and comes from the raw command (`res.gaze_out`,
        // lem_model.py:3128 and :4401; train_mimo.py:3244).
        let gaze = rawCommand.gazeAngles
        let fingerGoals = heldFingerGoals

        // 8. decode and write: 5 env steps of 2 physics steps
        let stepReaders = readers
        var controls: [MotorControls] = []
        controls.reserveCapacity(ModelStep.envStepsPerModelStep)
        for _ in 0 ..< ModelStep.envStepsPerModelStep {
            let envControls = try decoder.decode(sent, mask: handleMask, gaze: gaze, fingerGoals: fingerGoals)
            try physics.stepEnvStep(controls: envControls) { state in
                for reader in stepReaders {
                    try reader.observePhysicsStep(state)
                }
            }
            for reader in stepReaders {
                try reader.observeEnvStep(physics)
            }
            controls.append(envControls)
        }
        // The next model step's motor slot: the held command with the burst's gaze in its
        // eyes' row (train_mimo.py:3292-3294, gaze_to_handle).
        var executedForModel = sent
        executedForModel.setGaze(gaze)
        lastSentCommand = executedForModel
        lastControls = controls.last

        // 9. eye images
        var images: [EyeCamera: EyeImage] = [:]
        if eyeImageInterval > 0, step.count % eyeImageInterval == 0 {
            let geoms = try sceneGeoms.sceneGeoms()
            for camera in EyeCamera.allCases {
                let pose = try sceneGeoms.cameraPose(camera)
                images[camera] = try eyeRenderer.render(camera: camera, pose: pose, geoms: geoms)
            }
        }

        completedModelSteps += 1
        return ModelStepReport(
            step: step,
            readings: readings,
            inputs: inputs,
            output: output,
            planningEvent: outcome,
            rawCommand: rawCommand,
            executedCommand: executed,
            sentCommand: sent,
            latchedNewCommand: latched,
            controls: controls,
            eyeImages: images,
            goalRingCount: planner.goalRingCount)
    }

    /// The loop's own state now.
    public func snapshot() -> SessionSnapshot {
        SessionSnapshot(
            completedModelSteps: completedModelSteps,
            heldCommand: heldCommand,
            heldFingerGoals: heldFingerGoals,
            lastControls: lastControls,
            goalRingCount: planner.goalRingCount,
            planningEventsRun: planningEventsRun,
            planningEventsSkipped: planningEventsSkipped)
    }

    /// Resets every part and the loop state to model step 0.
    public func reset() throws {
        try physics.reset()
        for reader in readers {
            reader.reset()
        }
        model.reset()
        planner.reset()
        executor.reset()
        completedModelSteps = 0
        planningEventsRun = 0
        planningEventsSkipped = 0
        heldCommand = nil
        heldFingerGoals = nil
        lastSentCommand = nil
        lastControls = nil
    }
}

// MARK: - Saving the snapshot

/// Writes the session snapshot as one JSON file (keys sorted), replacing the previous one
/// atomically, and reads it back.
public final class JSONSessionStateStore: SessionStateSaving {
    /// The file the snapshot is written to.
    public let fileURL: URL

    /// A store writing to `fileURL`; its folder must exist.
    public init(fileURL: URL) {
        self.fileURL = fileURL
    }

    /// Writes `snapshot` to `fileURL`.
    public func save(_ snapshot: SessionSnapshot) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(snapshot)
        try data.write(to: fileURL, options: .atomic)
    }

    /// The saved snapshot; nil when no file exists.
    public func load() throws -> SessionSnapshot? {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            return nil
        }
        let data = try Data(contentsOf: fileURL)
        return try JSONDecoder().decode(SessionSnapshot.self, from: data)
    }
}

// MARK: - The view model

/// Starts, pauses and single-steps the session loop and publishes each model step's
/// results: the model-step count, the six tokens, the command sent to the decoder, the
/// last planning event's scores and the five eye images. Every completed model step's
/// report is passed to the injected observers (SensesViewModel, CommandViewModel,
/// PlannerViewModel, MemoryViewModel). On a background scene phase the loop stops and the
/// snapshot is saved. The model steps run on the main actor, one per loop turn, with a
/// `Task.yield()` between them so the screens update.
@MainActor
public final class RunViewModel: ObservableObject {
    /// Model steps completed.
    @Published public private(set) var completedModelSteps = 0

    /// True while the loop runs model steps on its own.
    @Published public private(set) var isRunning = false

    /// The six tokens of the last model step; empty before the first.
    @Published public private(set) var tokens: TokenSet = [:]

    /// The command the decoder read in the last model step (.centimetres); nil before the
    /// first.
    @Published public private(set) var command: Command? = nil

    /// The scores of the last planning event that ran, start first (21 for a full event);
    /// empty before the first.
    @Published public private(set) var plannerScores: [Float] = []

    /// The eye images of the last model step that rendered them.
    @Published public private(set) var eyeImages: [EyeCamera: EyeImage] = [:]

    /// The last model step's full report.
    @Published public private(set) var lastReport: ModelStepReport? = nil

    /// The error that stopped the loop; nil when none.
    @Published public private(set) var lastErrorDescription: String? = nil

    /// The snapshot written on the last background notice; nil when none.
    @Published public private(set) var lastSavedSnapshot: SessionSnapshot? = nil

    private let runner: any ModelStepRunning
    private let observers: [any ModelStepObserving]
    private let stateSaver: any SessionStateSaving
    private var loopTask: Task<Void, Never>?

    /// A view model over an injected runner.
    public init(runner: any ModelStepRunning, observers: [any ModelStepObserving], stateSaver: any SessionStateSaving) {
        self.runner = runner
        self.observers = observers
        self.stateSaver = stateSaver
        self.completedModelSteps = runner.completedModelSteps
    }

    /// A view model over a ModelStepPipeline made of the given parts (the parts' rules are
    /// those of `ModelStepPipeline.init`).
    public convenience init(
        physics: any Physics,
        readers: [any SenseReader],
        tokenizers: [any Tokenizer],
        model: any ForwardModeling,
        planner: any Planning,
        decoder: any CommandDecoding,
        executor: any CommandExecuting,
        eyeRenderer: any EyeRendering,
        sceneGeoms: any SceneGeomReading,
        planningPermit: any PlanningPermitting,
        observers: [any ModelStepObserving],
        stateSaver: any SessionStateSaving,
        handleMask: HandleMask = .allTrue
    ) throws {
        let pipeline = try ModelStepPipeline(
            physics: physics, readers: readers, tokenizers: tokenizers, model: model, planner: planner,
            decoder: decoder, executor: executor, eyeRenderer: eyeRenderer, sceneGeoms: sceneGeoms,
            planningPermit: planningPermit, handleMask: handleMask)
        self.init(runner: pipeline, observers: observers, stateSaver: stateSaver)
    }

    /// Runs one model step and publishes it; returns false (and stops the loop) when the
    /// model step threw.
    @discardableResult
    public func stepOnce() -> Bool {
        do {
            let report = try runner.runModelStep()
            publish(report)
            return true
        } catch {
            lastErrorDescription = String(describing: error)
            pause()
            return false
        }
    }

    /// Starts running model steps one after another until `pause()`, an error or a
    /// background notice. Does nothing while already running.
    public func start() {
        guard loopTask == nil else { return }
        lastErrorDescription = nil
        isRunning = true
        loopTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self = self, self.isRunning else { return }
                guard self.stepOnce() else { return }
                await Task.yield()
            }
        }
    }

    /// Stops the loop after the model step in progress.
    public func pause() {
        isRunning = false
        loopTask?.cancel()
        loopTask = nil
    }

    /// Reacts to the scene phase: `.background` stops the loop and saves the snapshot;
    /// `.active` and `.inactive` change nothing (the loop is restarted by `start()`).
    public func handleScenePhase(_ phase: ScenePhase) {
        if phase == .background {
            enterBackground()
        }
    }

    /// Stops the loop and writes the snapshot synchronously (inside the five seconds
    /// before suspension).
    public func enterBackground() {
        pause()
        let snapshot = runner.snapshot()
        do {
            try stateSaver.save(snapshot)
            lastSavedSnapshot = snapshot
        } catch {
            lastErrorDescription = String(describing: error)
        }
    }

    /// Stops the loop and resets the session to model step 0.
    public func reset() {
        pause()
        do {
            try runner.reset()
            lastErrorDescription = nil
        } catch {
            lastErrorDescription = String(describing: error)
        }
        completedModelSteps = runner.completedModelSteps
        tokens = [:]
        command = nil
        plannerScores = []
        eyeImages = [:]
        lastReport = nil
        for observer in observers {
            observer.reset()
        }
    }

    private func publish(_ report: ModelStepReport) {
        completedModelSteps = runner.completedModelSteps
        tokens = report.output.tokens
        command = report.sentCommand
        if case .ran(let plan) = report.planningEvent {
            plannerScores = plan.scores
        }
        if !report.eyeImages.isEmpty {
            eyeImages = report.eyeImages
        }
        lastReport = report
        for observer in observers {
            observer.observe(report)
        }
    }
}
