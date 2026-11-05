// OneModelStepTests.swift: the real stack for 20 model steps on macOS, recording peak resident
// memory. Work item WI-40, target B, wave 5 of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html (row WI-40: "The real stack for 20
// model steps on macOS, recording peak resident memory. Arithmetic bound today: 73.1 MB of
// weights plus one MjData arena of about 400 MB against the user-reported iPhone ceilings of
// section 10."; acceptance: "First build: 20 model steps complete; peak memory in MB into
// section 9; the iOS-device run unverified until done.").
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain, no Xcode; plan section 3, fact 1).
//
// WHAT THIS FILE HOLDS (one responsibility per type; every stage reached through a protocol,
// every concrete type injected):
// - `ResidentMemoryReading` and `DarwinResidentMemory`: the process's resident memory in
//   bytes, current (mach_task_basic_info.resident_size) and peak (the larger of
//   getrusage(RUSAGE_SELF).ru_maxrss, bytes on Darwin, and mach_task_basic_info.resident_size_max).
// - `CommandExecuting`: the execution transform from the raw command (scale `.normalized`)
//   to the executed command (scale `.centimetres`) that the decoder and the forward model's
//   motor slot read (planner_loss.md section 3 step 2: `candidate_execution`, lem_model.py:
//   6297-6314; Command.swift's CommandScale doc: "which is the session's step, not a
//   multiplication by 200").
// - `ExecutedCommandRecording`: where each decoded command is written for the motor reader
//   (MotorReader.swift: "A holder the session writes the executed command into after each
//   decode"; `ExecutedCommandRecord` conforms with no new code, its `record(_:mask:)` has this
//   signature).
// - `BodyFunctionModulating`: the body-function modulation run after each decode, before the
//   write (`body_function` "modulate", weights/b1_continue_B1K_skip13_492800/run/args.json:51;
//   env.py:2892-2893 as MotorFunction.swift's header quotes it; `MotorFunctionApplier.apply()`
//   conforms with no new code).
// - `GazeSourcing`: the gaze pair passed to the decoder. nil in the stack means the pair read
//   from the command's eye row (`Command.gazeAngles`), which is what the decoder does with
//   `gaze: nil`; the Python rewrites the pair with the saccade generator and the gaze reflex
//   first (env.py:2870-2874; `gaze_reflex on`, args.json:19), which no Swift file ports yet
//   (command_decoder.md open point A).
// - `ModelStepStack`: one existential per stage (Physics, the six SenseReaders, their
//   Tokenizers, ForwardModeling, Planning, CommandExecuting, CommandDecoding, the HandleMask,
//   and the three optional stages above).
// - `ModelStepLoop`: runs model steps over a stack in the sequence of plan section 2 and row
//   WI-32 ("per model step physics 10 steps, read, tokenize, model, command (planner every
//   10th), decode, write"), and returns one `ModelStepRecord` per model step.
// - `OneModelStepRunReport`: the 20-model-step result written as JSON, with the peak resident
//   memory in MB (1 MB = 1,000,000 bytes, the unit of the plan's "73.1 MB of weights" =
//   73,119,988 bytes, MANIFEST.json:120).
// - Fakes of every protocol, so the loop itself is tested before the real stack exists.
// - `ReleasedStackComposer` (compiled only with the condition UEMA_REAL_STACK): the
//   concrete types of WI-25 to WI-39 assembled into a `ModelStepStack`.
//
// THE SEQUENCE OF ONE MODEL STEP (model step count k = 1, 2, ...; `ModelStep.count` is "the
// count of model steps completed since the session started", ModelStep.swift):
//  1. Physics: 5 env steps (ModelStep.envStepsPerModelStep, args.json:154 `action_repeat` 5).
//     Before each env step, in the order of env.py:2884-2893 (MotorCommandWriter in
//     CommandDecoder.swift keeps the same order): the decoder turns the executed command into
//     90 controls (a fresh decode every env step: command_decoder.md section 1, quoted in
//     Protocols.swift's `stepEnvStep` doc) with the gaze pair of the `GazeSourcing` stage (nil:
//     the command's eye row); the executed command and the mask are written to the
//     `ExecutedCommandRecording` stage; the `BodyFunctionModulating` stage runs; then the write.
//     Model step 1 has no executed command yet and runs `MotorControls.zero` with no decode,
//     no record and no modulation. Each env step is 2 physics steps of 0.005 s (env.py:1157
//     `frame_skip=2`); every SenseReader is called after each physics step (the tactile read
//     per substep, env.py:2358-2360) and after each env step (the fatigue model and the
//     vestibular fields, proprio_rows.md section 3, vestibular_fields.md section 1).
//  2. Read: each SenseReader returns its `SenseReading` for model step k.
//  3. Tokenize: each sense's Tokenizer turns it into an `EncoderInput`.
//  4. Model: `ForwardModeling.step(inputs:previousExecutedCommand:at:)`, with the executed
//     command of model step k-1 (nil at k = 1; proprio_rows.md section 5 step 3).
//  5. Goal ring: `Planning.record(tokens:wellness:at:)` before the planning test, as the
//     Python appends to the ring (lem_model.py:4271) before it tests
//     `len(self.goal_ring) >= 20 and step % self.intention_every == 0` (lem_model.py:4283).
//     So the ring holds 20 entries at k = 20, and the one planning event of a 20-model-step
//     run is at k = 20 (at k = 10 the ring holds 10).
//  6. Planner: when `canPlan(at:)`, `plan(from:at:)` replaces the motor controller's command
//     with the selected raw command and latches the plan's finger joint goals for the decoder
//     until the next planning event (command_decoder.md section 1: the latched finger-joint goal).
//  7. Execute: `CommandExecuting.execute` gives the executed command (scale `.centimetres`),
//     decoded through the next model step's 5 env steps.
//
// WHERE THIS SEQUENCE DIFFERS FROM THE PYTHON (first-build reviewers check both):
//  a. The Python trainer reads the state, runs the model and then the 5 env steps; this loop
//     follows plan row WI-32 and SenseReader.read's contract ("The sense's raw arrays for the
//     model step that just completed", Protocols.swift), so physics runs first and model step 1
//     runs 10 physics steps of zero controls before the first read: the first reading is of the
//     state 0.050 s after the reset, not of the reset state.
//  b. The gaze pair: see `GazeSourcing` above. The released composer passes no gaze source.
//
// PASS LINES OF THE REAL-STACK TEST, declared before the first measurement:
//  - 20 model steps complete (plan row WI-40).
//  - nq = 146 (plan rows WI-15 and WI-25: "nq equals 146").
//  - simulated time advanced 20 x 0.050 s = 1.0 s within 1e-9 s (plan row WI-25: "simulated
//    time advanced 1.0 s"; the float64 rounding of 200 additions of 0.005 s stays under 1e-12 s).
//  - every qpos number, every token number and every predicted token number finite (plan row
//    WI-25: "no NaN in qpos"; row WI-35: "no NaN").
//  - every one of the 95 decoded control sets within -1 to 1 (row WI-37: "every control
//    within -1..1"; 19 model steps x 5 env steps, model step 1 runs zero controls).
//  - exactly 1 planning event, at model step 20 (sequence point 5 above), with 1 to 21 scores
//    (PlanResult.evaluationsPerEvent 21; fewer when a non-finite objective stopped the
//    search, planner_loss.md section 3 step 8), and the selected score at or below the start
//    score (row WI-39: "the chosen score at or below the start").
//  - peak resident memory below 3,000 MB: specification REQ-B-15's line ("below 3,000 MB with
//    the entitlement, the lowest ceiling reported"), applied here to the macOS run. The plan's
//    arithmetic for it: 73.1 MB of weights (73,119,988 bytes, MANIFEST.json:120) plus one MuJoCo
//    data arena of 380 MB (measured for the H84 world model, Source/Python/mimo_bridge/tools/
//    movement_viewer/commands.py: "the H84 world model's MjData reserves a 380 MB arena"), plus
//    the context 589,824 bytes and the goal ring 921,600 bytes (REQ-B-15): 454.6 MB before
//    MLX's activations, the planner's gradient passes and the eye render targets, which no
//    file measures. The device run stays unverified until it is done (plan row WI-40).
//
// FIRST BUILD ON A MAC (this work item's acceptance; plan section 8 steps 2 to 5):
//  1. Plan section 8 steps 1 to 3 first: the clone on branch feature/apple-port-2026-11-04,
//     `python fetch_weights.py --accept-license` (model.safetensors, 73,119,988 bytes, not in
//     git), `bash apple/scripts/build_mujoco_xcframework.sh` exit 0 (WI-18), `cd apple/UEMAKit
//     && swift package resolve` exit 0 (WI-17).
//  2. `xcodebuild -scheme UEMAKit-Package test -destination 'platform=macOS'
//     -only-testing:UEMAIntegrationTests` : exit 0. Without the condition UEMA_REAL_STACK the
//     released-stack test reports "skipped" with the reason printed, and the other 8 tests of
//     this file pass: the loop on fakes (20 model steps, 200 physics steps, 95 decodes, 1
//     planning event at model step 20, the per-env-step order decode, record, modulate,
//     write); the gaze source reaching the decoder; the planning cadence (events at 20, 30,
//     40, 50, 60 over 60 model steps); the refusal of a stack with no vestibular reader; the
//     refusal of an executed command at `.normalized` scale; the refusal of an unreadable
//     memory probe; the memory probe; the JSON record. Plus the placeholder test of
//     UEMAIntegrationPackageTests.swift.
//  3. Once WI-25, WI-27 to WI-31, WI-35 to WI-37 and WI-39 have landed: run the same command
//     with `OTHER_SWIFT_FLAGS='$(inherited) -D UEMA_REAL_STACK'` and the weights path in the
//     environment variable UEMA_WEIGHTS. xcodebuild hands the test process every variable
//     whose name starts with TEST_RUNNER_, prefix removed (Xcode behaviour outside the
//     evidence files: unverified until this step), so write
//     `TEST_RUNNER_UEMA_WEIGHTS=/path/to/model.safetensors xcodebuild ...`; `swift test`
//     passes UEMA_WEIGHTS as it is, but cannot build MLX's Metal shaders (Package.swift
//     header). A compile error in `ReleasedStackComposer` means a concrete type's name or
//     initializer differs from the names this file took from the plan (listed in that
//     type's doc comment): correct the composer to the landed type, never the type.
//  4. The test prints "WI-40 peak resident memory: <n> MB" and the path of the JSON record
//     (UEMA_RECORD_DIR, or the temporary folder when unset). Copy the number, the record's
//     path, the commit hash and the clock time (AM/PM Pacific) into the plan's section 9 row
//     WI-40 and into apple/FIRST_BUILD_RECORD.md (WI-41). A peak at or above 3,000 MB, or any
//     other pass line missed, reads FAILED in section 9.
//  5. The iOS-device run (plan section 8 step 7) stays unverified until done; its peak comes
//     from MemoryView (WI-38) on the device.
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave:
// - apple/UEMAKit/Sources/UEMACore/Protocols.swift (whole file): Physics with the default
//   `stepEnvStep` (write, then 2 physics steps, calling `afterEachPhysicsStep` after each);
//   SenseReader's observe defaults; Tokenizer; ForwardModeling.step; Planning.record, canPlan,
//   plan and the default `canPlan`; CommandDecoding.decode, which throws `wrongScale` for a
//   `.normalized` command; PlanResult's checks.
// - apple/UEMAKit/Sources/UEMACore/ModelStep.swift: envStepsPerModelStep 5,
//   physicsStepsPerEnvStep 2, physicsTimestepSeconds 0.005, seconds 0.050,
//   goalRingMinimumForPlanning 20, planningInterval 10, `isPlanningCandidate`.
// - apple/UEMAKit/Sources/UEMACore/Command.swift: Command (234 numbers, scale),
//   MotorControls (90, range -1 to 1, `isWithinRange`), HandleMask.allTrue, FingerJointGoals,
//   GazeAngles.
// - apple/UEMAKit/Sources/UEMACore/SenseName.swift: SenseName.active (six senses), Token
//   (192 numbers), TokenSet, `missingActiveSenses`.
// - apple/UEMAKit/Specs/planner_loss.md sections 1 and 3: the planning condition and the
//   execution transform `candidate_execution` (step 2), the 21 evaluations and the stop on a
//   non-finite objective (step 8).
// - apple/UEMAKit/Specs/command_decoder.md section 1 (the `gaze` row: the saccade generator
//   and the gaze reflex rewrite the pair before the decoder, env.py:2870-2874) and open point A.
// - apple/UEMAKit/Sources/UEMAScene/SceneBundle.swift: `ResourceSceneBundle()`, `sceneXMLURL`,
//   `verify()`. apple/UEMAKit/Sources/UEMAModel/SafetensorsLoader.swift:
//   `SafetensorsLoader.bundled()` and `load(from:)` (WeightsLoading).
// - apple/UEMAKit/Sources/UEMASenses/MotorReader.swift: `ExecutedCommandRecord.record(_:mask:)`
//   and `MotorReader(tracker:executed:)`. apple/UEMAKit/Sources/UEMACommand/MotorFunction.swift:
//   `MotorFunctionApplier(modulator:mode:capacities:writer:)` and `apply()`.
//   apple/UEMAKit/Sources/UEMACommand/CommandDecoder.swift: MotorCommandWriter's order "the
//   decode ..., then the body-function modulation ..., then the write".
//   apple/UEMAKit/Sources/UEMAPlanner/Planner.swift and GoalRing.swift:
//   `Planner(ring:goalRule:transitions:search:handDescent:)`, `GoalRing()`,
//   `HighestWellnessGoalSelection()`, `AdamCommandSearch()`.
//   apple/UEMAKit/Sources/UEMATokenizer/TokenizerBank.swift: `TokenizerBank.released(vestibular:)`
//   and its `tokenizers` dictionary. apple/UEMAKit/Sources/UEMAPhysics/MuJoCoPhysics.swift:
//   `MuJoCoPhysics()` and `load(sceneXML:)`.
// - weights/b1_continue_B1K_skip13_492800/run/args.json lines 19 (`"gaze_reflex": "on"`), 51
//   (`"body_function": "modulate"`), 62 (`"intention_every": 10`), 154 (`"action_repeat": 5`).
// - Source/Python/mimo_bridge/lem_model.py line 4271 (the ring append) and line 4283 (the
//   planning test), and docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html REQ-B-15 (the
//   3,000 MB line and the byte counts quoted above).
// - getrusage(2) on Darwin: ru_maxrss "the maximum resident set size utilized (in bytes)";
//   mach_task_basic_info's resident_size and resident_size_max through task_info with flavor
//   MACH_TASK_BASIC_INFO. Both are Apple-side facts outside the plan's evidence files, from
//   the Darwin headers' documentation as known to the writer, no page fetched: unverified
//   until the first build prints a peak.

import Darwin
import Foundation
import XCTest
import UEMACore
#if UEMA_REAL_STACK
import UEMACommand
import UEMAModel
import UEMAPhysics
import UEMAPlanner
import UEMAScene
import UEMASenses
import UEMATokenizer
#endif

// MARK: - Resident memory

/// Reads the resident memory of this process in bytes.
protocol ResidentMemoryReading {
    /// The resident memory now, in bytes; nil when the system call fails.
    func currentResidentBytes() -> Int?

    /// The largest resident memory since the process started, in bytes; nil when the system
    /// calls fail.
    func peakResidentBytes() -> Int?
}

/// Resident memory from Darwin: current from mach_task_basic_info.resident_size; peak as the
/// larger of getrusage(RUSAGE_SELF).ru_maxrss (bytes on Darwin) and
/// mach_task_basic_info.resident_size_max.
struct DarwinResidentMemory: ResidentMemoryReading {
    /// Bytes per MB in every record of this file: 1,000,000 (the plan's "73.1 MB" for
    /// 73,119,988 bytes).
    static let bytesPerMegabyte = 1_000_000.0

    func currentResidentBytes() -> Int? {
        basicInfo().map { Int($0.resident_size) }
    }

    func peakResidentBytes() -> Int? {
        var usage = rusage()
        let rusagePeak: Int? = getrusage(RUSAGE_SELF, &usage) == 0 ? Int(usage.ru_maxrss) : nil
        let machPeak: Int? = basicInfo().map { Int($0.resident_size_max) }
        switch (rusagePeak, machPeak) {
        case let (a?, b?):
            return max(a, b)
        case let (a?, nil):
            return a
        case let (nil, b?):
            return b
        case (nil, nil):
            return nil
        }
    }

    /// `bytes` in MB (1 MB = 1,000,000 bytes).
    static func megabytes(_ bytes: Int) -> Double {
        Double(bytes) / bytesPerMegabyte
    }

    private func basicInfo() -> mach_task_basic_info? {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result: kern_return_t = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { reboundPointer in
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), reboundPointer, &count)
            }
        }
        return result == KERN_SUCCESS ? info : nil
    }
}

// MARK: - The stages the UEMACore contract does not name

/// The execution transform: the raw command (scale `.normalized`, the motor controller's or
/// the planner's) into the executed command (scale `.centimetres`) the decoder reads and the
/// forward model's motor slot takes on the next model step (planner_loss.md section 3 step
/// 2, `candidate_execution`, lem_model.py:6297-6314: the step limit around the achieved
/// handle positions, the strength limit and the 0.35 smoothing blend).
protocol CommandExecuting {
    /// The executed command of `raw`, reading the achieved handle state from `physics`.
    func execute(_ raw: Command, physics: any PhysicsReading) throws -> Command
}

/// Where each decoded command and its mask are written for the motor reader, once per env
/// step after the decode (MotorReader.swift, `ExecutedCommandRecord`, whose
/// `record(_:mask:)` has this signature).
protocol ExecutedCommandRecording: AnyObject {
    /// Records the command decoded for this env step and its mask.
    func record(_ command: Command, mask: HandleMask?)
}

/// The body-function modulation of actuator gain and force range, once per env step after
/// the decode and before the write (env.py:2892-2893 under `body_function` "modulate",
/// args.json:51; `MotorFunctionApplier.apply()` has this signature).
protocol BodyFunctionModulating: AnyObject {
    /// Writes the modulated actuator parameters of the current body-set capacities.
    func apply() throws
}

/// The gaze pair the decoder takes for one env step (command_decoder.md section 1, `gaze`).
protocol GazeSourcing: AnyObject {
    /// The azimuth and elevation in radians for decoding `command` in the state `physics` holds.
    func gaze(for command: Command, physics: any PhysicsReading) throws -> GazeAngles
}

// MARK: - The stack and the loop

/// Errors of the model-step loop.
enum ModelStepLoopError: Error, Equatable {
    /// No tokenizer was given for a sense that has a reader.
    case missingTokenizer(SenseName)
    /// No reader was given for an active sense.
    case missingReader(SenseName)
    /// Two readers were given for one sense.
    case duplicateReader(SenseName)
    /// A command of the wrong scale left a stage.
    case wrongScale(stage: String, expected: CommandScale, found: CommandScale)
    /// The memory probe returned no reading.
    case memoryUnreadable
}

/// One existential per stage of plan section 2, injected into `ModelStepLoop`.
struct ModelStepStack {
    /// The MuJoCo physics, scene already loaded.
    let physics: any Physics
    /// One reader per active sense, in any order.
    let readers: [any SenseReader]
    /// One tokenizer per active sense.
    let tokenizers: [SenseName: any Tokenizer]
    /// Encoders, transformer, heads, motor controller and wellness.
    let forwardModel: any ForwardModeling
    /// The goal ring and the planner.
    let planner: any Planning
    /// The raw command into the executed command.
    let executor: any CommandExecuting
    /// The executed command into 90 controls.
    let decoder: any CommandDecoding
    /// The decoder's handle mask: all 26 true in the released run (lem_model.py:5692-5693).
    let mask: HandleMask
    /// Where each decoded command goes for the motor reader; nil records nothing.
    let executedCommandRecord: (any ExecutedCommandRecording)?
    /// The body-function modulation after each decode; nil runs none (`body_function off`).
    let bodyFunction: (any BodyFunctionModulating)?
    /// The gaze pair per decode; nil passes `gaze: nil`, the command's eye row.
    let gazeSource: (any GazeSourcing)?

    /// A stack of the given stages; the last three default to nil.
    init(
        physics: any Physics,
        readers: [any SenseReader],
        tokenizers: [SenseName: any Tokenizer],
        forwardModel: any ForwardModeling,
        planner: any Planning,
        executor: any CommandExecuting,
        decoder: any CommandDecoding,
        mask: HandleMask,
        executedCommandRecord: (any ExecutedCommandRecording)? = nil,
        bodyFunction: (any BodyFunctionModulating)? = nil,
        gazeSource: (any GazeSourcing)? = nil
    ) {
        self.physics = physics
        self.readers = readers
        self.tokenizers = tokenizers
        self.forwardModel = forwardModel
        self.planner = planner
        self.executor = executor
        self.decoder = decoder
        self.mask = mask
        self.executedCommandRecord = executedCommandRecord
        self.bodyFunction = bodyFunction
        self.gazeSource = gazeSource
    }

    /// Checks that every active sense has exactly one reader and a tokenizer.
    func validate() throws {
        var seen = Set<SenseName>()
        for reader in readers {
            guard !seen.contains(reader.sense) else {
                throw ModelStepLoopError.duplicateReader(reader.sense)
            }
            seen.insert(reader.sense)
            guard tokenizers[reader.sense] != nil else {
                throw ModelStepLoopError.missingTokenizer(reader.sense)
            }
        }
        for sense in SenseName.active where !seen.contains(sense) {
            throw ModelStepLoopError.missingReader(sense)
        }
    }
}

/// What one model step of the loop produced and measured.
struct ModelStepRecord {
    /// The model step (count after it completed).
    let step: ModelStep
    /// Simulated seconds the physics advanced during this model step (0.050 expected).
    let simulatedSecondsAdvanced: Double
    /// Physics steps run during this model step (10 expected).
    let physicsSteps: Int
    /// Decoded control sets during this model step (5, or 0 on model step 1).
    let decodes: Int
    /// True when every decoded control lay within -1 to 1.
    let controlsWithinRange: Bool
    /// True when every qpos number after the physics was finite.
    let qposFinite: Bool
    /// True when all six tokens held 192 finite numbers.
    let tokensFinite: Bool
    /// True when all six predicted tokens held 192 finite numbers.
    let predictedTokensFinite: Bool
    /// The active senses without a token in the model's output.
    let missingTokenSenses: [SenseName]
    /// True when the forward model was given no previous executed command.
    let previousCommandWasNil: Bool
    /// The planning event of this model step; nil when none ran.
    let plan: PlanResult?
    /// The wellness channel sum the model gave this model step.
    let wellness: Float
    /// The process's peak resident memory after this model step, bytes.
    let peakResidentBytes: Int
}

/// Runs model steps over a `ModelStepStack` in the sequence of plan section 2 (this file's
/// header, points 1 to 7). Holds the executed command and the latched finger joint goals
/// between model steps.
final class ModelStepLoop {
    private let stack: ModelStepStack
    private let memory: any ResidentMemoryReading

    /// The executed command of the last model step (scale `.centimetres`); nil before the first.
    private(set) var executedCommand: Command?

    /// The finger joint goals of the last planning event; nil before the first.
    private(set) var fingerGoals: FingerJointGoals?

    /// The model steps completed.
    private(set) var step = ModelStep.start

    /// A loop over `stack`, reading memory through `memory`; throws when a sense lacks a
    /// reader or tokenizer.
    init(stack: ModelStepStack, memory: any ResidentMemoryReading) throws {
        try stack.validate()
        self.stack = stack
        self.memory = memory
    }

    /// Runs `count` model steps and returns one record per model step.
    func run(modelSteps count: Int) throws -> [ModelStepRecord] {
        var records: [ModelStepRecord] = []
        records.reserveCapacity(count)
        for _ in 0 ..< count {
            records.append(try runModelStep())
        }
        return records
    }

    /// Runs one model step. A thrown error leaves `step`, `executedCommand` and `fingerGoals`
    /// as they were before the call (the physics and the readers have advanced).
    func runModelStep() throws -> ModelStepRecord {
        let next = step.advanced()
        let physics = stack.physics
        let readers = stack.readers
        let timeBefore = physics.time

        // 1. Physics: 5 env steps of 2 physics steps; before each, decode, record, modulate
        // (env.py:2884-2893), then the write inside stepEnvStep.
        var physicsSteps = 0
        var decodes = 0
        var controlsWithinRange = true
        for _ in 0 ..< ModelStep.envStepsPerModelStep {
            let controls: MotorControls
            if let executed = executedCommand {
                let gaze = try stack.gazeSource?.gaze(for: executed, physics: physics)
                controls = try stack.decoder.decode(executed, mask: stack.mask, gaze: gaze, fingerGoals: fingerGoals)
                stack.executedCommandRecord?.record(executed, mask: stack.mask)
                try stack.bodyFunction?.apply()
                decodes += 1
                controlsWithinRange = controlsWithinRange && controls.isWithinRange
            } else {
                controls = MotorControls.zero
            }
            try physics.stepEnvStep(controls: controls) { physicsNow in
                physicsSteps += 1
                for reader in readers {
                    try reader.observePhysicsStep(physicsNow)
                }
            }
            for reader in readers {
                try reader.observeEnvStep(physics)
            }
        }
        let qposFinite = physics.qpos.allSatisfy { $0.isFinite }

        // 2. Read and 3. tokenize.
        var inputs: [SenseName: EncoderInput] = [:]
        for reader in readers {
            let reading = try reader.read(physics, at: next)
            guard let tokenizer = stack.tokenizers[reading.sense] else {
                throw ModelStepLoopError.missingTokenizer(reading.sense)
            }
            inputs[reading.sense] = try tokenizer.tokenize(reading)
        }

        // 4. Model, with the executed command of the previous model step.
        let previousCommandWasNil = executedCommand == nil
        let output = try stack.forwardModel.step(inputs: inputs, previousExecutedCommand: executedCommand, at: next)
        guard output.command.scale == .normalized else {
            throw ModelStepLoopError.wrongScale(stage: "motor controller", expected: .normalized, found: output.command.scale)
        }

        // 5. Goal ring, before the planning test (lem_model.py:4271, :4283).
        stack.planner.record(tokens: output.tokens, wellness: output.wellness, at: next)

        // 6. Planner on its cadence.
        var raw = output.command
        var plan: PlanResult?
        if stack.planner.canPlan(at: next) {
            let result = try stack.planner.plan(from: output.command, at: next)
            raw = result.command
            plan = result
        }

        // 7. Execution transform; decoded during the next model step.
        let executed = try stack.executor.execute(raw, physics: physics)
        guard executed.scale == .centimetres else {
            throw ModelStepLoopError.wrongScale(stage: "execution transform", expected: .centimetres, found: executed.scale)
        }
        guard let peak = memory.peakResidentBytes() else {
            throw ModelStepLoopError.memoryUnreadable
        }

        // Every check passed: commit the state carried to the next model step.
        executedCommand = executed
        if let goals = plan?.fingerGoals {
            fingerGoals = goals
        }
        step = next

        return ModelStepRecord(
            step: next,
            simulatedSecondsAdvanced: physics.time - timeBefore,
            physicsSteps: physicsSteps,
            decodes: decodes,
            controlsWithinRange: controlsWithinRange,
            qposFinite: qposFinite,
            tokensFinite: ModelStepLoop.allFinite(output.tokens),
            predictedTokensFinite: ModelStepLoop.allFinite(output.predictedTokens),
            missingTokenSenses: output.tokens.missingActiveSenses,
            previousCommandWasNil: previousCommandWasNil,
            plan: plan,
            wellness: output.wellness,
            peakResidentBytes: peak)
    }

    /// True when every token of `tokens` holds 192 finite numbers.
    static func allFinite(_ tokens: TokenSet) -> Bool {
        tokens.values.allSatisfy { token in
            token.values.count == Token.count && token.values.allSatisfy { $0.isFinite }
        }
    }
}

// MARK: - The record

/// The result of one run, written as JSON for plan section 9 and apple/FIRST_BUILD_RECORD.md.
struct OneModelStepRunReport: Codable, Equatable {
    /// One planning event: its model step, its scores (start first), the selected index.
    struct PlanningEvent: Codable, Equatable {
        let modelStep: Int
        let scores: [Float]
        let selectedIteration: Int
    }

    /// Which stack ran: "fake" or "released".
    let stack: String
    /// Model steps completed.
    let modelSteps: Int
    /// Physics steps run.
    let physicsSteps: Int
    /// Decoded control sets.
    let decodes: Int
    /// Simulated seconds the physics advanced over the run.
    let simulatedSeconds: Double
    /// True when every pass condition of the records held: qpos, tokens and predicted tokens
    /// finite, every control within -1 to 1, no active sense without a token.
    let everyRecordFiniteAndInRange: Bool
    /// The planning events.
    let planningEvents: [PlanningEvent]
    /// Resident memory before the stack was built, MB (1 MB = 1,000,000 bytes); nil when unread.
    let residentBeforeStackMegabytes: Double?
    /// Peak resident memory of the process after the last model step, MB.
    let peakResidentMegabytes: Double
    /// The line the peak is judged against, MB: 3,000 (specification REQ-B-15).
    let peakLineMegabytes: Double

    /// The report of `records`.
    init(stack: String, records: [ModelStepRecord], residentBeforeStackBytes: Int?) {
        self.stack = stack
        modelSteps = records.count
        physicsSteps = records.reduce(0) { $0 + $1.physicsSteps }
        decodes = records.reduce(0) { $0 + $1.decodes }
        simulatedSeconds = records.reduce(0) { $0 + $1.simulatedSecondsAdvanced }
        everyRecordFiniteAndInRange = records.allSatisfy {
            $0.qposFinite && $0.tokensFinite && $0.predictedTokensFinite && $0.controlsWithinRange
                && $0.missingTokenSenses.isEmpty
        }
        planningEvents = records.compactMap { record in
            record.plan.map {
                PlanningEvent(modelStep: record.step.count, scores: $0.scores, selectedIteration: $0.selectedIteration)
            }
        }
        residentBeforeStackMegabytes = residentBeforeStackBytes.map(DarwinResidentMemory.megabytes)
        peakResidentMegabytes = DarwinResidentMemory.megabytes(records.map(\.peakResidentBytes).max() ?? 0)
        peakLineMegabytes = OneModelStepTests.peakLineMegabytes
    }

    /// Writes the report as JSON into UEMA_RECORD_DIR, or the temporary folder when unset, and
    /// returns the file's URL.
    func write(environment: [String: String] = ProcessInfo.processInfo.environment) throws -> URL {
        let folder: URL
        if let path = environment["UEMA_RECORD_DIR"], !path.isEmpty {
            folder = URL(fileURLWithPath: path, isDirectory: true)
        } else {
            folder = FileManager.default.temporaryDirectory
        }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent("wi40_one_model_step_\(stack).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(self).write(to: url)
        return url
    }
}

// MARK: - The tests

/// The model-step loop over fakes, the memory probe, and the released stack for 20 model
/// steps (plan row WI-40).
final class OneModelStepTests: XCTestCase {
    /// Model steps per run: 20 (plan row WI-40).
    static let modelSteps = 20

    /// Specification REQ-B-15's line for peak memory, MB.
    static let peakLineMegabytes = 3_000.0

    /// Decodes per 20-model-step run: model steps 2 to 20, 5 env steps each.
    static let expectedDecodes = (modelSteps - 1) * ModelStep.envStepsPerModelStep

    // MARK: The loop over fakes

    func testFakeStackRunsTwentyModelStepsInPlanSequence() throws {
        let fakes = FakeStack()
        let loop = try ModelStepLoop(stack: fakes.stack, memory: DarwinResidentMemory())
        let records = try loop.run(modelSteps: Self.modelSteps)

        XCTAssertEqual(records.count, 20)
        XCTAssertEqual(records.map(\.step.count), Array(1 ... 20))
        XCTAssertEqual(loop.step.count, 20)
        XCTAssertEqual(records.reduce(0) { $0 + $1.physicsSteps }, 200)
        XCTAssertEqual(fakes.physics.physicsStepCount, 200)
        XCTAssertEqual(fakes.physics.time, 1.0, accuracy: 1e-9)
        XCTAssertEqual(records.reduce(0) { $0 + $1.decodes }, Self.expectedDecodes)
        XCTAssertEqual(fakes.decoder.decodeCount, 95)
        XCTAssertEqual(records.first?.decodes, 0)
        XCTAssertTrue(records.allSatisfy { $0.controlsWithinRange && $0.qposFinite })
        XCTAssertTrue(records.allSatisfy { $0.tokensFinite && $0.predictedTokensFinite })
        XCTAssertTrue(records.allSatisfy { $0.missingTokenSenses.isEmpty })
        // The fake decoder sets every control to the wrist_L strength. Model step 20's env steps
        // decode model step 19's executed command (the motor controller's strength 0.25); the
        // planned command (strength 0.5) is first decoded on model step 21.
        XCTAssertEqual(fakes.physics.control[0], Double(FakeForwardModel.controllerStrength), accuracy: 1e-6)

        // Every reader called after each of the 200 physics steps and each of the 100 env
        // steps, and read once per model step.
        for reader in fakes.readers {
            XCTAssertEqual(reader.physicsStepCount, 200, "\(reader.sense)")
            XCTAssertEqual(reader.envStepCount, 100, "\(reader.sense)")
            XCTAssertEqual(reader.readCount, 20, "\(reader.sense)")
        }

        // Per env step: decode, record, modulate, write (env.py:2884-2893); model step 1
        // writes zero controls 5 times with no decode.
        let expectedEvents: [String] = Array(repeating: "write", count: ModelStep.envStepsPerModelStep)
            + Array(repeating: ["decode", "record", "modulate", "write"], count: Self.expectedDecodes).flatMap { $0 }
        XCTAssertEqual(fakes.log.events, expectedEvents)
        XCTAssertEqual(fakes.executedRecord.recordCount, 95)
        XCTAssertEqual(fakes.bodyFunction.applyCount, 95)
        XCTAssertEqual(fakes.executedRecord.lastMask, HandleMask.allTrue)
        XCTAssertNil(fakes.decoder.lastGaze, "no gaze source: the decoder reads the command's eye row")

        // The previous executed command: nil on model step 1 only, centimetres after.
        XCTAssertEqual(records.map(\.previousCommandWasNil), [true] + Array(repeating: false, count: 19))
        XCTAssertEqual(fakes.forwardModel.previousScales.compactMap { $0 }, [CommandScale](repeating: .centimetres, count: 19))

        // One planning event, at model step 20, when the ring first holds 20 entries.
        let planned = records.filter { $0.plan != nil }.map(\.step.count)
        XCTAssertEqual(planned, [20])
        XCTAssertEqual(fakes.planner.planSteps, [20])
        XCTAssertEqual(fakes.planner.goalRingCount, 20)
        let plan = try XCTUnwrap(records.last?.plan)
        XCTAssertEqual(plan.scores.count, PlanResult.evaluationsPerEvent)
        XCTAssertLessThanOrEqual(plan.selectedScore, plan.scores[0])

        // The planner's command became the executed command: after model step 20 its strength
        // is the planner's 0.5, not the motor controller's 0.25; the record of model step 20's
        // decodes still holds model step 19's executed command (strength 0.25).
        let executed = try XCTUnwrap(loop.executedCommand)
        XCTAssertEqual(executed.scale, .centimetres)
        XCTAssertEqual(executed.strength(of: .wristL), FakePlanner.plannedStrength)
        XCTAssertEqual(loop.fingerGoals, FakePlanner.fingerGoals)
        XCTAssertEqual(fakes.executedRecord.lastCommand?.strength(of: .wristL), FakeForwardModel.controllerStrength)
    }

    func testGazeSourceReachesDecoder() throws {
        let fakes = FakeStack()
        let gaze = FakeGazeSource()
        let loop = try ModelStepLoop(stack: fakes.makeStack(gazeSource: gaze), memory: DarwinResidentMemory())
        _ = try loop.run(modelSteps: 2)
        XCTAssertEqual(gaze.callCount, ModelStep.envStepsPerModelStep, "one call per decode of model step 2")
        XCTAssertEqual(fakes.decoder.lastGaze, FakeGazeSource.angles)
    }

    func testPlanningCadenceMatchesPythonCondition() {
        // lem_model.py:4283 after the append of 4271: ring length after model step k is k (up
        // to 200), so a planning event runs at k when k >= 20 and k % 10 == 0.
        let planner = FakePlanner()
        var planned: [Int] = []
        for k in 1 ... 60 {
            let step = ModelStep(count: k)
            planner.record(tokens: FakeForwardModel.tokens(value: 0), wellness: 0, at: step)
            if planner.canPlan(at: step) {
                planned.append(k)
            }
        }
        XCTAssertEqual(planned, [20, 30, 40, 50, 60])
    }

    func testLoopRefusesStackWithoutEverySense() throws {
        let fakes = FakeStack()
        let withoutVestibular = ModelStepStack(
            physics: fakes.physics,
            readers: fakes.readers.filter { $0.sense != .vestibular },
            tokenizers: fakes.stack.tokenizers,
            forwardModel: fakes.forwardModel,
            planner: fakes.planner,
            executor: FakeExecutor(),
            decoder: fakes.decoder,
            mask: .allTrue)
        XCTAssertThrowsError(try ModelStepLoop(stack: withoutVestibular, memory: DarwinResidentMemory())) { error in
            XCTAssertEqual(error as? ModelStepLoopError, .missingReader(.vestibular))
        }
    }

    func testLoopRefusesExecutedCommandAtNormalizedScale() throws {
        let fakes = FakeStack()
        let stack = fakes.makeStack(executor: FakeExecutor(outputScale: .normalized))
        let loop = try ModelStepLoop(stack: stack, memory: DarwinResidentMemory())
        XCTAssertThrowsError(try loop.runModelStep()) { error in
            XCTAssertEqual(
                error as? ModelStepLoopError,
                .wrongScale(stage: "execution transform", expected: .centimetres, found: .normalized))
        }
        XCTAssertEqual(loop.step, ModelStep.start, "a refused model step is not counted")
        XCTAssertNil(loop.executedCommand, "a refused command is not carried to the next model step")
    }

    func testLoopRefusesUnreadableMemory() throws {
        let fakes = FakeStack()
        let loop = try ModelStepLoop(stack: fakes.stack, memory: UnreadableMemory())
        XCTAssertThrowsError(try loop.runModelStep()) { error in
            XCTAssertEqual(error as? ModelStepLoopError, .memoryUnreadable)
        }
        XCTAssertEqual(loop.step, ModelStep.start)
        XCTAssertNil(loop.executedCommand)
    }

    // MARK: The memory probe and the record

    func testResidentMemoryProbeReadsThisProcess() throws {
        let probe = DarwinResidentMemory()
        let current = try XCTUnwrap(probe.currentResidentBytes())
        let peak = try XCTUnwrap(probe.peakResidentBytes())
        XCTAssertGreaterThan(current, 0)
        // Read current first: the peak read after it is at or above it.
        XCTAssertGreaterThanOrEqual(peak, current)
        XCTAssertEqual(DarwinResidentMemory.megabytes(73_119_988), 73.119988, accuracy: 1e-9)
    }

    func testReportOfFakeRunIsWrittenAsJSON() throws {
        let fakes = FakeStack()
        let probe = DarwinResidentMemory()
        let before = probe.currentResidentBytes()
        let loop = try ModelStepLoop(stack: fakes.stack, memory: probe)
        let records = try loop.run(modelSteps: Self.modelSteps)
        let report = OneModelStepRunReport(stack: "fake", records: records, residentBeforeStackBytes: before)
        XCTAssertEqual(report.modelSteps, 20)
        XCTAssertEqual(report.physicsSteps, 200)
        XCTAssertEqual(report.decodes, 95)
        XCTAssertEqual(report.simulatedSeconds, 1.0, accuracy: 1e-9)
        XCTAssertTrue(report.everyRecordFiniteAndInRange)
        XCTAssertEqual(report.planningEvents.map(\.modelStep), [20])
        XCTAssertEqual(report.planningEvents.first?.scores.count, PlanResult.evaluationsPerEvent)
        XCTAssertGreaterThan(report.peakResidentMegabytes, 0)
        XCTAssertEqual(report.peakLineMegabytes, 3_000)

        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("wi40_report_test_\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let url = try report.write(environment: ["UEMA_RECORD_DIR": folder.path])
        XCTAssertEqual(url.lastPathComponent, "wi40_one_model_step_fake.json")
        let decoded = try JSONDecoder().decode(OneModelStepRunReport.self, from: Data(contentsOf: url))
        XCTAssertEqual(decoded, report)
    }

    // MARK: The released stack (plan row WI-40)

    func testReleasedStackRunsTwentyModelSteps() throws {
        #if UEMA_REAL_STACK
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["UEMA_WEIGHTS"], !path.isEmpty else {
            throw XCTSkip("UEMA_WEIGHTS is unset: set it to model.safetensors (73,119,988 bytes, python fetch_weights.py --accept-license); with xcodebuild write TEST_RUNNER_UEMA_WEIGHTS")
        }
        let weightsURL = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw XCTSkip("UEMA_WEIGHTS names \(weightsURL.path), which does not exist")
        }

        let probe = DarwinResidentMemory()
        let before = probe.currentResidentBytes()
        let stack = try ReleasedStackComposer().makeStack(weightsURL: weightsURL)
        XCTAssertEqual(stack.physics.sizes.nq, 146, "plan rows WI-15 and WI-25: nq equals 146")
        XCTAssertEqual(stack.physics.sizes.nu, MotorControls.count)

        let loop = try ModelStepLoop(stack: stack, memory: probe)
        let records = try loop.run(modelSteps: Self.modelSteps)
        let report = OneModelStepRunReport(stack: "released", records: records, residentBeforeStackBytes: before)
        let url = try report.write()
        print("WI-40 peak resident memory: \(String(format: "%.1f", report.peakResidentMegabytes)) MB (line \(Int(Self.peakLineMegabytes)) MB, specification REQ-B-15); before the stack: \(report.residentBeforeStackMegabytes.map { String(format: "%.1f", $0) } ?? "unread") MB; record: \(url.path)")

        XCTAssertEqual(report.modelSteps, 20)
        XCTAssertEqual(report.physicsSteps, 200)
        XCTAssertEqual(report.simulatedSeconds, 1.0, accuracy: 1e-9)
        XCTAssertEqual(report.decodes, Self.expectedDecodes)
        XCTAssertTrue(records.allSatisfy(\.qposFinite), "a qpos number was not finite")
        XCTAssertTrue(records.allSatisfy(\.tokensFinite), "a token number was not finite")
        XCTAssertTrue(records.allSatisfy(\.predictedTokensFinite), "a predicted token number was not finite")
        XCTAssertTrue(records.allSatisfy(\.controlsWithinRange), "a control left -1 to 1")
        XCTAssertTrue(records.allSatisfy { $0.missingTokenSenses.isEmpty }, "an active sense had no token")

        XCTAssertEqual(report.planningEvents.map(\.modelStep), [20])
        let plan = try XCTUnwrap(records.last?.plan)
        XCTAssertTrue((1 ... PlanResult.evaluationsPerEvent).contains(plan.scores.count))
        XCTAssertLessThanOrEqual(plan.selectedScore, plan.scores[0], "plan row WI-39: the chosen score at or below the start")

        XCTAssertLessThan(report.peakResidentMegabytes, Self.peakLineMegabytes, "specification REQ-B-15: peak below 3,000 MB")
        #else
        throw XCTSkip("compiled without UEMA_REAL_STACK: build with OTHER_SWIFT_FLAGS='$(inherited) -D UEMA_REAL_STACK' once WI-25, WI-27 to WI-31, WI-35 to WI-37 and WI-39 have landed (this file's header, first-build step 3)")
        #endif
    }
}

// MARK: - The released stack

/// Builds a `ModelStepStack`.
protocol ModelStepStackComposing {
    /// The stack over the weights at `weightsURL`.
    func makeStack(weightsURL: URL) throws -> ModelStepStack
}

#if UEMA_REAL_STACK
/// `ExecutedCommandRecord.record(_:mask:)` (MotorReader.swift) has the requirement's signature.
extension ExecutedCommandRecord: ExecutedCommandRecording {}

/// `MotorFunctionApplier.apply()` (MotorFunction.swift) has the requirement's signature.
extension MotorFunctionApplier: BodyFunctionModulating {}

/// The released stack from the concrete types of plan rows WI-25 to WI-39.
///
/// Names and initializers read on disk when this was written: `ResourceSceneBundle()`,
/// `verify()`, `sceneXMLURL` (UEMAScene/SceneBundle.swift); `MuJoCoPhysics()` and
/// `load(sceneXML:)` (UEMAPhysics/MuJoCoPhysics.swift); `SafetensorsLoader.bundled()` and
/// `load(from:)` (UEMAModel/SafetensorsLoader.swift); `ExecutedCommandRecord()`
/// (UEMASenses/MotorReader.swift); `TokenizerBank.released(vestibular:)` and `tokenizers`
/// (UEMATokenizer/TokenizerBank.swift); `Planner(ring:goalRule:transitions:search:handDescent:)`,
/// `GoalRing()`, `HighestWellnessGoalSelection()`, `AdamCommandSearch()` (UEMAPlanner);
/// `MotorFunctionApplier(modulator:mode:capacities:writer:)` (UEMACommand/MotorFunction.swift).
///
/// Names NOT on disk at writing, taken from the plan's rows: correct each to the landed type at
/// the first build (this file's header, step 3):
/// - `ForwardModel(weights:)` (WI-35, WI-36; ForwardModel.swift), conforming to
///   `ForwardModeling` and to UEMAPlanner's `PlanTransitionProviding` (Planner.swift: "the
///   session's forward model holds the context captured before this model step's shift").
/// - `ReleasedSenseReaders.make(physics:scene:executed:)` returning `.readers` (one reader per
///   active sense: ProprioceptionReader, VestibularReader, MotorReader over `executed` of WI-27,
///   TactileSense of WI-28, the vision reader over SceneKitEyeRenderer of WI-29, the audio
///   reader) and `.tracker`, the one shared `BodyStateTracker`.
/// - `ReleasedVestibularNormaliser.make(weights:)`: the checkpoint's frozen mean and variance
///   (vestibular_fields.md section 3) into `VestibularNormaliser(mean:variance:weight:)`.
/// - `ReleasedHandDescent.make(physics:)`: the hand descent of planner_loss.md section 4
///   (`JointHandDescent`, WI-39).
/// - `MotorFunctionModulator.released(physics:)`: `MotorFunctionModulator(tables:)` over the
///   loaded model's actuator tables. The applier also needs `BodyStateTracker:
///   BodySetCapacityReading` and `MuJoCoPhysics: ActuatorStrengthWriting` (MotorFunction.swift's
///   first-build steps 2 and 3).
/// - `CommandDecoder.released(physics:)` (WI-37; the landed initializer is
///   `CommandDecoder(engine:physics:fingerJointIDs:)`).
/// - `CandidateExecution(physics:)`: the execution transform (planner_loss.md section 3 step 2),
///   owned by no plan row; it needs `extension CandidateExecution: CommandExecuting {}` here.
// State on November 4, 2026 (consistency pass, apple/FIRST_BUILD_CHECKLIST.md section 3): the
// body-function lines below use the adapters that ReleasedSceneCommandTests runs on the released
// scene, and the forward model is built by ForwardModel.released(from:). Still not on disk, so
// this block does not compile yet: ForwardModel's conformances to UEMACore.ForwardModeling (its
// step returns UEMAModel.ForwardPass, not ModelOutput, and runs neither the motor controller nor
// wellness) and to UEMAPlanner.PlanTransitionProviding; ReleasedSenseReaders.make (which would
// build the six readers, now including UEMASenses' VisionReader and AudioReader, and the shared
// BodyStateTracker); ReleasedVestibularNormaliser.make; ReleasedHandDescent.make;
// CandidateExecution (the execution transform, lem_model.py:5664-5727).
struct ReleasedStackComposer: ModelStepStackComposing {
    func makeStack(weightsURL: URL) throws -> ModelStepStack {
        let scene = try ResourceSceneBundle()
        _ = try scene.verify()
        let physics = MuJoCoPhysics()
        try physics.load(sceneXML: scene.sceneXMLURL)
        let loaded = try SafetensorsLoader.bundled().load(from: weightsURL)
        // The tables before any body-function write (ActuatedBodyTables(modelTables:) doc).
        let actuated = try ActuatedBodyTables(modelTables: physics.modelTables())
        let forwardModel = try ForwardModel.released(from: loaded.weights)
        let executedRecord = ExecutedCommandRecord()
        let senses = try ReleasedSenseReaders.make(physics: physics, scene: scene, executed: executedRecord)
        let tokenizers = TokenizerBank.released(
            vestibular: try ReleasedVestibularNormaliser.make(weights: loaded.weights)).tokenizers
        let planner = Planner(
            ring: GoalRing(),
            goalRule: HighestWellnessGoalSelection(),
            transitions: forwardModel,
            search: AdamCommandSearch(),
            handDescent: try ReleasedHandDescent.make(physics: physics))
        let bodyFunction = MotorFunctionApplier(
            modulator: try MotorFunctionModulator(tables: actuated),
            mode: .modulate,
            capacities: FatigueCapacities(source: senses.tracker),
            writer: ActuatorParameterStrengthWriter(physics))
        return ModelStepStack(
            physics: physics,
            readers: senses.readers,
            tokenizers: tokenizers,
            forwardModel: forwardModel,
            planner: planner,
            executor: CandidateExecution(physics: physics),
            decoder: try CommandDecoder.released(physics: physics),
            mask: .allTrue,
            executedCommandRecord: executedRecord,
            bodyFunction: bodyFunction,
            gazeSource: nil)
    }
}
#endif

// MARK: - Fakes

/// The calls of the per-env-step stages in the sequence they happened ("decode", "record",
/// "modulate", "write").
final class CallLog {
    private(set) var events: [String] = []

    func append(_ event: String) {
        events.append(event)
    }
}

/// Every fake of one fake stack, kept by concrete type so a test reads their counters.
final class FakeStack {
    let log: CallLog
    let physics: FakePhysics
    let readers: [FakeSenseReader]
    let forwardModel: FakeForwardModel
    let planner: FakePlanner
    let decoder: FakeDecoder
    let executedRecord: FakeExecutedRecord
    let bodyFunction: FakeBodyFunction

    init() {
        let log = CallLog()
        self.log = log
        physics = FakePhysics(log: log)
        readers = SenseName.active.map { FakeSenseReader(sense: $0) }
        forwardModel = FakeForwardModel()
        planner = FakePlanner()
        decoder = FakeDecoder(log: log)
        executedRecord = FakeExecutedRecord(log: log)
        bodyFunction = FakeBodyFunction(log: log)
        // The fake is loaded on creation; the scene URL is not read.
        try? physics.load(sceneXML: URL(fileURLWithPath: "/dev/null"))
    }

    /// The stack over these fakes with the default executor and no gaze source.
    var stack: ModelStepStack { makeStack() }

    /// The stack over these fakes with `executor` and `gazeSource`.
    func makeStack(executor: any CommandExecuting = FakeExecutor(), gazeSource: (any GazeSourcing)? = nil) -> ModelStepStack {
        var tokenizers: [SenseName: any Tokenizer] = [:]
        for sense in SenseName.active {
            tokenizers[sense] = FakeTokenizer(sense: sense)
        }
        return ModelStepStack(
            physics: physics,
            readers: readers,
            tokenizers: tokenizers,
            forwardModel: forwardModel,
            planner: planner,
            executor: executor,
            decoder: decoder,
            mask: .allTrue,
            executedCommandRecord: executedRecord,
            bodyFunction: bodyFunction,
            gazeSource: gazeSource)
    }
}

/// A physics with the released model's sizes (`PhysicsSizes.released`: nq 146, nv 136, nu 90,
/// nbody 76, njnt 104, nsensordata 168, ngeom 188, ncam 6; ngeom and ncam are unread by the loop)
/// whose time advances 0.005 s per physics step and whose qpos[0] follows control[0].
final class FakePhysics: Physics {
    let sizes = PhysicsSizes.released
    private(set) var time: Double = 0
    private(set) var qpos: [Double] = Array(repeating: 0, count: 146)
    private(set) var qvel: [Double] = Array(repeating: 0, count: 136)
    let sensordata: [Double] = Array(repeating: 0, count: 168)
    let xpos: [Double] = Array(repeating: 0, count: 76 * 3)
    let xquat: [Double] = Array(repeating: 0, count: 76 * 4)
    let xmat: [Double] = Array(repeating: 0, count: 76 * 9)
    let xaxis: [Double] = Array(repeating: 0, count: 104 * 3)
    let xanchor: [Double] = Array(repeating: 0, count: 104 * 3)
    let actuatorForce: [Double] = Array(repeating: 0, count: 90)
    private(set) var control: [Double] = Array(repeating: 0, count: 90)
    let contacts: [Contact] = []
    private(set) var isLoaded = false
    private(set) var physicsStepCount = 0
    private let log: CallLog?

    init(log: CallLog? = nil) {
        self.log = log
    }

    func id(of kind: MuJoCoObjectKind, named name: String) -> Int? { nil }

    func bodyVelocity(bodyId: Int) -> [Double] { Array(repeating: 0, count: 6) }

    func load(sceneXML url: URL) throws { isLoaded = true }

    func reset() throws {
        time = 0
        qpos = Array(repeating: 0, count: sizes.nq)
        qvel = Array(repeating: 0, count: sizes.nv)
    }

    func setState(qpos: [Double], qvel: [Double]) throws {
        guard qpos.count == sizes.nq else {
            throw UEMACoreError.wrongCount(type: "FakePhysics.qpos", expected: sizes.nq, found: qpos.count)
        }
        guard qvel.count == sizes.nv else {
            throw UEMACoreError.wrongCount(type: "FakePhysics.qvel", expected: sizes.nv, found: qvel.count)
        }
        self.qpos = qpos
        self.qvel = qvel
    }

    func write(controls: MotorControls) throws {
        log?.append("write")
        control = controls.clipped().values
    }

    func stepPhysicsStep() throws {
        qpos[0] += 0.001 * control[0]
        time += ModelStep.physicsTimestepSeconds
        physicsStepCount += 1
    }
}

/// A reader that counts the calls it receives and returns a 4-number zero reading.
final class FakeSenseReader: SenseReader {
    let sense: SenseName
    private(set) var physicsStepCount = 0
    private(set) var envStepCount = 0
    private(set) var readCount = 0

    init(sense: SenseName) {
        self.sense = sense
    }

    func observePhysicsStep(_ physics: any PhysicsReading) throws {
        physicsStepCount += 1
    }

    func observeEnvStep(_ physics: any PhysicsReading) throws {
        envStepCount += 1
    }

    func read(_ physics: any PhysicsReading, at step: ModelStep) throws -> SenseReading {
        readCount += 1
        return SenseReading(sense: sense, primary: ShapedArray(zerosOfShape: [4]))
    }

    func reset() {
        physicsStepCount = 0
        envStepCount = 0
        readCount = 0
    }
}

/// A tokenizer that passes the reading's array through.
struct FakeTokenizer: Tokenizer {
    let sense: SenseName

    func tokenize(_ reading: SenseReading) throws -> EncoderInput {
        guard reading.sense == sense else {
            throw ModelStepLoopError.missingTokenizer(reading.sense)
        }
        return EncoderInput(sense: sense, primary: reading.primary, auxiliary: reading.auxiliary, isFresh: reading.isFresh)
    }
}

/// A forward model whose tokens are all 0.01 x the model step count, whose command has
/// strength 0.25 on every handle, and whose wellness is the count modulo 7.
final class FakeForwardModel: ForwardModeling {
    /// The strength of every handle of the motor controller's command.
    static let controllerStrength: Float = 0.25

    let contextLength = ModelStep.contextLength
    private(set) var filledContextSteps = 0
    /// The scale of each previous executed command given, nil where none was given.
    private(set) var previousScales: [CommandScale?] = []

    /// Six tokens whose 192 numbers all equal `value`.
    static func tokens(value: Float) -> TokenSet {
        var set: TokenSet = [:]
        for sense in SenseName.active {
            set[sense] = Token(repeating: value)
        }
        return set
    }

    func step(inputs: [SenseName: EncoderInput], previousExecutedCommand: Command?, at step: ModelStep) throws -> ModelOutput {
        for sense in SenseName.active where inputs[sense] == nil {
            throw UEMACoreError.missingSense(sense)
        }
        previousScales.append(previousExecutedCommand?.scale)
        filledContextSteps = min(filledContextSteps + 1, contextLength)
        var command = Command.zero(scale: .normalized)
        for handle in BodyHandle.allCases {
            command[handle, .strength] = FakeForwardModel.controllerStrength
        }
        let value = Float(step.count) * 0.01
        return ModelOutput(
            step: step,
            tokens: FakeForwardModel.tokens(value: value),
            predictedTokens: FakeForwardModel.tokens(value: value + 0.01),
            command: command,
            wellness: Float(step.count % 7))
    }

    func reset() {
        filledContextSteps = 0
        previousScales = []
    }
}

/// A planner over a goal ring of the last 200 entries whose plan returns strength 0.5 on
/// every handle, 21 decreasing scores and the last one selected.
final class FakePlanner: Planning {
    /// The strength of every handle of the planned command.
    static let plannedStrength: Float = 0.5

    /// The finger joint goals every plan returns: 44 angles of 0.1 rad.
    static let fingerGoals: FingerJointGoals? = try? FingerJointGoals(angles: Array(repeating: 0.1, count: FingerJointGoals.count))

    private var ring: [(step: ModelStep, wellness: Float)] = []
    private(set) var planSteps: [Int] = []

    var goalRingCount: Int { ring.count }

    func record(tokens: TokenSet, wellness: Float, at step: ModelStep) {
        ring.append((step: step, wellness: wellness))
        if ring.count > ModelStep.goalRingLength {
            ring.removeFirst(ring.count - ModelStep.goalRingLength)
        }
    }

    func plan(from command: Command, at step: ModelStep) throws -> PlanResult {
        planSteps.append(step.count)
        var planned = command
        for handle in BodyHandle.allCases {
            planned[handle, .strength] = FakePlanner.plannedStrength
        }
        let goal = ring.max { $0.wellness < $1.wellness } ?? (step: step, wellness: 0)
        let scores = (0 ..< PlanResult.evaluationsPerEvent).map { Float(PlanResult.evaluationsPerEvent - $0) }
        return try PlanResult(
            command: planned,
            scores: scores,
            selectedIteration: PlanResult.evaluationsPerEvent - 1,
            goalStep: goal.step,
            goalWellness: goal.wellness,
            fingerGoals: FakePlanner.fingerGoals)
    }

    func reset() {
        ring = []
        planSteps = []
    }
}

/// The fake execution transform: positions x 200 (centimetres per position_range unit,
/// Command.positionRangeCentimetres), every other column copied, tagged `outputScale`
/// (`.centimetres` by default; `.normalized` copies the positions unscaled, to test the loop's
/// refusal). Not the released transform, which steps from the achieved handle positions and
/// blends (planner_loss.md section 3 step 2).
struct FakeExecutor: CommandExecuting {
    var outputScale: CommandScale = .centimetres

    func execute(_ raw: Command, physics: any PhysicsReading) throws -> Command {
        guard raw.scale == .normalized else {
            throw UEMACoreError.wrongScale(expected: .normalized, found: raw.scale)
        }
        guard outputScale == .centimetres else {
            return try Command(values: raw.values, scale: outputScale)
        }
        var values = raw.values
        for handle in BodyHandle.allCases {
            for column in CommandColumn.position {
                values[Command.index(handle: handle, column: column)] *= Command.positionRangeCentimetres
            }
        }
        return try Command(values: values, scale: .centimetres)
    }
}

/// A decoder that refuses a `.normalized` command, as the contract says, and returns every
/// control at the command's wrist_L strength, clipped to -1 to 1.
final class FakeDecoder: CommandDecoding {
    let motorContract = "ancestor_arm_axes_signed_limits_v3+finger_joint_servo_v1"
    private(set) var decodeCount = 0
    /// The gaze pair of the last decode; nil when none was passed.
    private(set) var lastGaze: GazeAngles?
    private let log: CallLog?

    init(log: CallLog? = nil) {
        self.log = log
    }

    func decode(_ command: Command, mask: HandleMask, gaze: GazeAngles?, fingerGoals: FingerJointGoals?) throws -> MotorControls {
        guard command.scale == .centimetres else {
            throw UEMACoreError.wrongScale(expected: .centimetres, found: command.scale)
        }
        log?.append("decode")
        decodeCount += 1
        lastGaze = gaze
        let level = Double(command.strength(of: .wristL))
        return try MotorControls(values: Array(repeating: level, count: MotorControls.count)).clipped()
    }
}

/// A record of the decoded commands that counts them and keeps the last one.
final class FakeExecutedRecord: ExecutedCommandRecording {
    private let log: CallLog
    private(set) var recordCount = 0
    private(set) var lastCommand: Command?
    private(set) var lastMask: HandleMask?

    init(log: CallLog) {
        self.log = log
    }

    func record(_ command: Command, mask: HandleMask?) {
        log.append("record")
        recordCount += 1
        lastCommand = command
        lastMask = mask
    }
}

/// A body-function modulation that counts its calls.
final class FakeBodyFunction: BodyFunctionModulating {
    private let log: CallLog
    private(set) var applyCount = 0

    init(log: CallLog) {
        self.log = log
    }

    func apply() throws {
        log.append("modulate")
        applyCount += 1
    }
}

/// A gaze source that returns azimuth 0.1 rad and elevation -0.2 rad and counts its calls.
final class FakeGazeSource: GazeSourcing {
    static let angles = GazeAngles(azimuth: 0.1, elevation: -0.2)
    private(set) var callCount = 0

    func gaze(for command: Command, physics: any PhysicsReading) throws -> GazeAngles {
        callCount += 1
        return FakeGazeSource.angles
    }
}

/// A memory probe whose system calls fail.
struct UnreadableMemory: ResidentMemoryReading {
    func currentResidentBytes() -> Int? { nil }

    func peakResidentBytes() -> Int? { nil }
}
