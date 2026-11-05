// UEMAMindApp.swift: the entry point and composition root of UEMA Mind, the SwiftUI app of
// the Apple port. Work item WI-38, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; this is the work item's main
// file. Its views are in apple/UEMAMind/Views/ (RunControlView, EyesView, SensesView,
// CommandView, PlannerView, MemoryView, WeightsPickerView); its view models are WI-32's,
// in apple/UEMAMind/ViewModels/ (RunViewModel, SensesViewModel, CommandViewModel,
// PlannerViewModel and MemoryViewModel, each read for this file).
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). SwiftUI only (plan section 3, fact 4); no UIKit or
// AppKit type (specification T-B-7).
//
// TYPES IN THIS FILE, one responsibility each:
//   SessionBuilding           protocol: makes the RunViewModel of a session from a weights
//                             file, a planning permit and the panel observers
//   RealStackSessionBuilder   SessionBuilding for the released stack; today it throws,
//                             since the concrete physics, model, planner and decoder
//                             (WI-25, WI-35, WI-36, WI-37, WI-39) are built by their own
//                             items and not wired into this app, and the SessionFactory of
//                             WI-32 is not on disk (see "WHAT IS MISSING" below)
//   MindAppModel              ObservableObject: the app's state before and after a session
//                             exists (the builder, WI-32's four panel view models, the
//                             session)
//   MindRootView, SessionScreen   the screen: the seven views over the view models
//   UEMAMindApp               the @main App
//   (WeightsPickerView.swift holds WeightsFileStoring and ApplicationSupportWeightsStore,
//   the kept copy of the weights file that MindAppModel is given)
//   FakeStackSessionBuilder, PreviewFake* (DEBUG only)
//                             a fake stack of every protocol the loop takes, so one model
//                             step runs in a preview without MuJoCo, MLX or weights; its
//                             numbers are synthetic, never the model's
//
// THE SEQUENCE. Launch: MindAppModel holds a MemoryViewModel over ProcInfoMemoryMonitor
// (MemoryViewModel.swift:58-79, 107-119) and no session; the run buttons are disabled.
// Pick model.safetensors (WeightsPickerView): MindAppModel.pickWeights copies it into
// <Application Support>/UEMAMind and checks its SHA-256 against weight_manifest.json's
// (ApplicationSupportWeightsStore, WeightsPickerView.swift); on later launches
// loadKeptWeights() starts from that copy. Then the builder makes a RunViewModel with the
// MemoryViewModel as its planning permit (a due planning event is skipped at thermal state
// serious or critical, MemoryViewModel.swift:121-171) and with four observers
// (MemoryViewModel, SensesViewModel, CommandViewModel, PlannerViewModel), each told every
// model step's ModelStepReport (RunViewModel.swift, protocol ModelStepObserving and struct
// ModelStepReport). "One model step" calls
// RunViewModel.stepOnce; "Start" and "Pause" call start and pause. A background scene
// phase is passed to RunViewModel.handleScenePhase, which stops the loop and saves the
// snapshot.
//
// WHAT IS MISSING (stated in the work item's result, not built here): the wiring of the
// real stack's concrete types into a session (MuJoCoPhysics WI-25, the forward model WI-35
// and WI-36, the command decoder WI-37, the planner WI-39, the execution transform behind
// WI-32's `CommandExecuting`; the first four are written by their own items at the same
// time as this file, with interfaces this file does not read) and the SessionFactory the
// plan's "Names" paragraph gives to WI-32. Also outside this item's file list: the plan
// row's SceneView and SettingsView.
// RealStackSessionBuilder is the one place the integration commit replaces with that
// factory; until then picking the weights file reports which parts are missing, and the
// DEBUG build offers "Run on the fake stack".
//
// FIRST BUILD ON A MAC (this work item's acceptance: "the macOS app launches; one model
// step runs on fake physics in a preview; after WI-40, 10 on the real stack"):
//  1. Run WI-17's, WI-18's and WI-33's steps first: the MuJoCo XCFramework in place,
//     `cd apple/UEMAKit && swift package resolve` exit 0, `cd apple && xcodegen generate`
//     exit 0 (apple/project.yml compiles every file under apple/UEMAMind except Tests/).
//  2. `cd apple && xcodebuild -scheme UEMAMind-macOS -destination 'platform=macOS' build`
//     : exit 0 (plan section 8, the step naming WI-33 and WI-38). A compile error that names
//     a RunViewModel or MemoryViewModel member means this file is out of date with
//     apple/UEMAMind/ViewModels/, and the view models win.
//  3. Open apple/UEMAMind.xcodeproj in Xcode, open UEMAMindApp.swift, show the canvas, and
//     select the preview "One model step on the fake stack" (it runs one model step when
//     it appears): it shows "1 model step, 0.05 s of simulated time (50 ms each)", five
//     eye images and six token grids. That is the "one model step runs on fake physics in
//     a preview" line.
//  4. Run UEMAMind-macOS (Product > Run): the window opens with the run buttons disabled;
//     in a Debug build press "Run on the fake stack", then "One model step" ten times: the
//     counter reads "10 model steps" and the five eye images shift by 8 pixels per press;
//     the planner panel still reads "No planning event yet" (the goal ring holds 10 of
//     the 20 entries a planning event needs, and the count must be a multiple of 10,
//     ModelStep.swift:63-71); the 21st press runs model step 20, the first planning event.
//  5. `xcodebuild -scheme UEMAMind-macOS -destination 'platform=macOS' test` : exit 0 with
//     every test of apple/UEMAMind/Tests/ViewsTests.swift passing (19 tests) beside WI-32's
//     RunViewModelTests. The tests import the app module as `UEMAMind_macOS`; a "no such
//     module" error means the module name differs: read PRODUCT_MODULE_NAME in the
//     generated project and correct the import.
//  6. `xcodebuild -scheme UEMAMind-iOS -destination 'generic/platform=iOS' build` : exit 0
//     (needs a signing team, plan section 8); `grep -rnE
//     '[U]IView|[N]SView|[U]IImage|[N]SImage|[U]IHosting|[N]SHosting' apple/UEMAMind`
//     prints nothing (specification T-B-7: "grep 0 UIKit or AppKit views"; the bracket in
//     each name keeps this line from matching itself).
//  7. Constructs to check first if step 2 fails, each used on purpose: `@MainActor` on the
//     App struct and on the views that create view models; `.onChange(of:initial:_:)` with
//     the two-value closure (macOS 14, iOS 17); `Image(decorative:scale:)` over a CGImage
//     of 24 bits per pixel with CGImageAlphaInfo.none (EyesView.swift); `Canvas`
//     (SensesView.swift); `Grid` and `GridRow` (CommandView.swift); `#Preview` closures
//     that declare a value and `return` the view; CryptoKit's `SHA256()` with
//     `update(data:)` and `finalize()`, and `FileHandle.read(upToCount:)` (macOS 10.15.4,
//     iOS 13.4) in WeightsPickerView.swift.
//  8. After WI-40: replace RealStackSessionBuilder with WI-32's SessionFactory, pick
//     model.safetensors, press "One model step" ten times: the counter reads 10 and the
//     five eye images change (plan section 8). Picking a file whose SHA-256 differs from
//     8f99a3ce...a315a (weight_manifest.json:7) shows "The picked file's SHA-256 is ...;
//     the released model.safetensors has ..." and keeps no copy.
//  8a. On the iPhone (plan row WI-38: "the weights file loads from Application Support
//     after a relaunch"): pick the file once, quit the app, launch it again: the session
//     is made from <Application Support>/UEMAMind/model.safetensors without the picker
//     (before WI-40 the message names the parts to wire instead, which also shows the
//     kept copy was found).
//  9. Record each step's command, exit code and clock time (AM/PM Pacific) in
//     apple/FIRST_BUILD_RECORD.md (WI-41).
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave:
// - work_items.json entry WI-38 and plan row WI-38 (quoted in each view's header);
//   acceptance "First build: the macOS app launches; one model step runs on fake physics in
//   a preview; after WI-40, 10 on the real stack."
// - docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html REQ-B-14 (the screens), REQ-B-15
//   (the 3,000 MB footprint line), T-B-7 ("Previews compile on both platforms; grep 0 UIKit
//   or AppKit views").
// - apple/UEMAMind/ViewModels/RunViewModel.swift (WI-32): RunViewModel's published
//   completedModelSteps, isRunning, tokens, command, plannerScores, eyeImages, lastReport,
//   lastErrorDescription; its convenience init over physics, readers, tokenizers, model,
//   planner, decoder, executor, eyeRenderer, sceneGeoms, planningPermit, observers,
//   stateSaver; ModelStepReport, PlanningEventOutcome, CommandExecuting, PlanningPermitting,
//   ModelStepObserving, JSONSessionStateStore.
// - apple/UEMAMind/ViewModels/MemoryViewModel.swift (WI-32): MemoryViewModel(monitor:),
//   availableMemoryMB, peakFootprintMB, thermalState; ProcInfoMemoryMonitor().
// - apple/UEMAMind/ViewModels/SensesViewModel.swift, CommandViewModel.swift,
//   PlannerViewModel.swift (WI-32): each `init()` and ModelStepObserving; rows and step;
//   sentRows, latchedThisModelStep and step; goalRingCount, scores, selectedIteration,
//   goalStep, goalWellness, lastEventStep and eventsSkipped.
// - apple/UEMAKit/Sources/UEMACore/Protocols.swift, Command.swift, ModelStep.swift,
//   SenseName.swift (WI-20): every protocol the fake stack implements, with its value types.
// - apple/project.yml (WI-33): the module names UEMAMind_macOS and UEMAMind_iOS follow
//   from PRODUCT_NAME = the target name ("PRODUCT_NAME: $(TARGET_NAME)").

import Combine
import Foundation
import SwiftUI
import UEMACore

// MARK: - Making a session

/// Makes the RunViewModel of one session. The concrete parts are built here and injected
/// into the view model, never constructed inside it.
@MainActor
protocol SessionBuilding {
    /// A name for the screen, such as "released stack" or "fake stack".
    var name: String { get }

    /// The session of the weights at `weightsURL` (nil for a builder that reads no weights),
    /// with `planningPermit` deciding whether a due planning event runs and `observers`
    /// told every model step's report.
    func makeRunViewModel(
        weightsURL: URL?,
        planningPermit: any PlanningPermitting,
        observers: [any ModelStepObserving]
    ) throws -> RunViewModel
}

/// Why the released stack cannot be made yet.
enum SessionBuildingError: Error, Equatable, CustomStringConvertible {
    /// The concrete types the session needs are not part of the build.
    case partsNotBuilt([String])

    /// A builder that reads weights was given no file.
    case noWeightsFile

    var description: String {
        switch self {
        case .partsNotBuilt(let parts):
            return "The released stack is not wired into this build; parts to wire: " + parts.joined(separator: ", ")
        case .noWeightsFile:
            return "No weights file was picked."
        }
    }
}

/// The released stack. Throws until the integration commit replaces it with WI-32's
/// SessionFactory over MuJoCoPhysics (WI-25), the six readers and tokenizers (WI-27 to
/// WI-30), the forward model (WI-35, WI-36), the planner (WI-39), the command decoder
/// (WI-37) and the eye renderer (WI-29).
struct RealStackSessionBuilder: SessionBuilding {
    /// The parts whose concrete types this build does not hold.
    static let missingParts = [
        "MuJoCoPhysics (WI-25)",
        "the forward model, motor controller and wellness (WI-35, WI-36)",
        "the command decoder (WI-37)",
        "the planner (WI-39)",
        "the execution transform behind CommandExecuting (WI-32)",
        "SessionFactory (WI-32)",
    ]

    var name: String { "released stack" }

    func makeRunViewModel(
        weightsURL: URL?,
        planningPermit: any PlanningPermitting,
        observers: [any ModelStepObserving]
    ) throws -> RunViewModel {
        guard weightsURL != nil else {
            throw SessionBuildingError.noWeightsFile
        }
        throw SessionBuildingError.partsNotBuilt(RealStackSessionBuilder.missingParts)
    }
}

// MARK: - App state

/// The app's state: the session builder, the four panel view models of WI-32 (the memory
/// view model alive from launch, so the launch reading of available memory is taken at
/// launch) and the session once one exists.
@MainActor
final class MindAppModel: ObservableObject {
    /// The session; nil until a builder made one.
    @Published private(set) var run: RunViewModel?

    /// The last message of a session build: the builder's name on success, the error
    /// otherwise.
    @Published private(set) var loadMessage: String?

    /// The memory and thermal readings, and the planning permit of every session.
    let memory: MemoryViewModel

    /// The sense panel's numbers.
    let senses: SensesViewModel

    /// The command table.
    let command: CommandViewModel

    /// The planner panel's numbers.
    let planner: PlannerViewModel

    private let builder: any SessionBuilding

    /// The kept, checked copy of the weights file; nil passes the picked file straight to
    /// the builder (previews and tests).
    private let weightsStore: (any WeightsFileStoring)?

    /// An app model over `builder` and `memory`, keeping picked weights in `weightsStore`.
    init(builder: any SessionBuilding, memory: MemoryViewModel, weightsStore: (any WeightsFileStoring)? = nil) {
        self.builder = builder
        self.memory = memory
        self.weightsStore = weightsStore
        self.senses = SensesViewModel()
        self.command = CommandViewModel()
        self.planner = PlannerViewModel()
    }

    /// The file dialog's pick: with a store, copies the file into it, checks its SHA-256
    /// and makes the session from the kept copy; a refused file keeps the current session
    /// and reports why. Without a store, makes the session from `url` itself. Runs while
    /// the dialog's security scope is held (WeightsPickerView).
    func pickWeights(_ url: URL) {
        guard let store = weightsStore else {
            makeSession(weightsURL: url)
            return
        }
        do {
            let kept = try store.importWeights(from: url)
            makeSession(weightsURL: kept)
        } catch {
            loadMessage = String(describing: error)
        }
    }

    /// At launch: makes the session from the kept copy when the store holds one and no
    /// session exists yet; does nothing otherwise.
    func loadKeptWeights() {
        guard run == nil, let store = weightsStore, let kept = store.storedWeightsURL() else { return }
        makeSession(weightsURL: kept)
    }

    /// Makes a session with `builder`; on failure keeps no session and reports the error.
    func makeSession(weightsURL: URL?, with builder: (any SessionBuilding)? = nil) {
        let chosen = builder ?? self.builder
        run?.pause()
        senses.reset()
        command.reset()
        planner.reset()
        do {
            run = try chosen.makeRunViewModel(
                weightsURL: weightsURL,
                planningPermit: memory,
                observers: [memory, senses, command, planner])
            loadMessage = "Session ready: \(chosen.name)"
        } catch {
            run = nil
            loadMessage = String(describing: error)
        }
    }
}

// MARK: - The screen

/// The screen before and after a session exists.
@MainActor
struct MindRootView: View {
    @ObservedObject var app: MindAppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("UEMA Mind").font(.title2.bold())
                WeightsPickerView(loadMessage: app.loadMessage) { url in
                    app.pickWeights(url)
                }
                #if DEBUG
                Button("Run on the fake stack (synthetic numbers, no weights)") {
                    app.makeSession(weightsURL: nil, with: FakeStackSessionBuilder())
                }
                #endif
                if let run = app.run {
                    SessionScreen(
                        run: run, memory: app.memory, senses: app.senses, command: app.command,
                        planner: app.planner)
                } else {
                    RunControlView(
                        modelStep: .start, isRunning: false, canRun: false,
                        onStart: {}, onPause: {}, onStepOnce: {})
                    MemoryScreen(memory: app.memory)
                }
            }
            .padding()
        }
        #if os(macOS)
        .frame(minWidth: 720, minHeight: 640)
        #endif
    }
}

/// The memory view over the memory view model.
@MainActor
struct MemoryScreen: View {
    @ObservedObject var memory: MemoryViewModel

    var body: some View {
        MemoryView(
            availableMegabytes: memory.availableMemoryMB,
            peakFootprintMegabytes: memory.peakFootprintMB,
            thermalState: memory.thermalState)
    }
}

/// The seven views of one session over its view models.
@MainActor
struct SessionScreen: View {
    @ObservedObject var run: RunViewModel
    let memory: MemoryViewModel
    @ObservedObject var senses: SensesViewModel
    @ObservedObject var command: CommandViewModel
    @ObservedObject var planner: PlannerViewModel
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            RunControlView(
                modelStep: ModelStep(count: run.completedModelSteps),
                isRunning: run.isRunning,
                canRun: true,
                onStart: { run.start() },
                onPause: { run.pause() },
                onStepOnce: { _ = run.stepOnce() })
            if let error = run.lastErrorDescription {
                Text("The loop stopped: \(error)")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
            EyesView(images: run.eyeImages, renderedAt: renderedAt)
            SensesView(rows: senses.rows, step: senses.step)
            PlannerView(
                goalRingCount: planner.goalRingCount,
                scores: planner.scores,
                selectedIteration: planner.selectedIteration,
                goalStep: planner.goalStep,
                goalWellness: planner.goalWellness,
                lastEventStep: planner.lastEventStep,
                eventsSkipped: planner.eventsSkipped)
            CommandView(
                rows: command.sentRows,
                positionUnit: CommandViewModel.positionUnit(of: .centimetres),
                caption: commandCaption)
            MemoryScreen(memory: memory)
        }
        .onChange(of: scenePhase) { _, newPhase in
            run.handleScenePhase(newPhase)
        }
    }

    /// The model step of the latest eye render when the last model step rendered; nil
    /// otherwise (EyesView then labels the images "the latest render").
    private var renderedAt: ModelStep? {
        guard let report = run.lastReport, !report.eyeImages.isEmpty else { return nil }
        return report.step
    }

    private var commandCaption: String {
        guard let step = command.step else { return "none yet" }
        let held = command.latchedThisModelStep ? "latched at this model step" : "held from an earlier model step"
        return "sent to the decoder at model step \(step.count), \(held)"
    }
}

// MARK: - The app

/// UEMA Mind: one MIMo body, the released forward model and planner, on macOS 14 and
/// iOS 17 or newer.
@main
@MainActor
struct UEMAMindApp: App {
    @StateObject private var app = MindAppModel(
        builder: RealStackSessionBuilder(),
        memory: MemoryViewModel(monitor: ProcInfoMemoryMonitor()),
        weightsStore: ApplicationSupportWeightsStore())

    var body: some Scene {
        WindowGroup {
            MindRootView(app: app)
                .onAppear { app.loadKeptWeights() }
        }
    }
}

// MARK: - The fake stack (DEBUG only)

#if DEBUG
/// A SessionBuilding over the fake stack below: every number is synthetic. It exists so
/// that one model step runs in a preview and in a Debug build without MuJoCo, MLX or the
/// weights file (this work item's acceptance).
struct FakeStackSessionBuilder: SessionBuilding {
    var name: String { "fake stack (synthetic numbers)" }

    func makeRunViewModel(
        weightsURL: URL?,
        planningPermit: any PlanningPermitting,
        observers: [any ModelStepObserving]
    ) throws -> RunViewModel {
        let physics = PreviewFakePhysics()
        let eyes = PreviewFakeEyes()
        let store = JSONSessionStateStore(
            fileURL: FileManager.default.temporaryDirectory.appendingPathComponent("uema_mind_fake_session.json"))
        return try RunViewModel(
            physics: physics,
            readers: SenseName.active.map { PreviewFakeSenseReader(sense: $0) },
            tokenizers: SenseName.active.map { PreviewFakeTokenizer(sense: $0) },
            model: PreviewFakeForwardModel(),
            planner: PreviewFakePlanner(),
            decoder: PreviewFakeDecoder(),
            executor: PreviewFakeExecutor(),
            eyeRenderer: eyes,
            sceneGeoms: eyes,
            planningPermit: planningPermit,
            observers: observers,
            stateSaver: store)
    }
}

/// Physics with the released sizes (nq 146, nv 136, nu 92, nbody 76, njnt 97, nsensordata
/// 56; Protocols.swift:242-245) and no dynamics beyond time and one coordinate: each
/// physics step adds 0.005 s and moves qpos[0] by 0.001 x the mean control.
final class PreviewFakePhysics: Physics {
    let sizes = PhysicsSizes(nq: 146, nv: 136, nu: 92, nbody: 76, njnt: 97, nsensordata: 56, ngeom: 0, ncam: 5)
    private(set) var time: Double = 0
    private(set) var qpos: [Double] = Array(repeating: 0, count: 146)
    private(set) var qvel: [Double] = Array(repeating: 0, count: 136)
    var sensordata: [Double] { Array(repeating: 0, count: sizes.nsensordata) }
    var xpos: [Double] { Array(repeating: 0, count: sizes.nbody * 3) }
    var xquat: [Double] { (0 ..< sizes.nbody).flatMap { _ in [1.0, 0, 0, 0] } }
    var xmat: [Double] { (0 ..< sizes.nbody).flatMap { _ in [1.0, 0, 0, 0, 1, 0, 0, 0, 1] } }
    var xaxis: [Double] { Array(repeating: 0, count: sizes.njnt * 3) }
    var xanchor: [Double] { Array(repeating: 0, count: sizes.njnt * 3) }
    var actuatorForce: [Double] { Array(repeating: 0, count: sizes.nu) }
    private(set) var control: [Double] = Array(repeating: 0, count: 92)
    var contacts: [Contact] { [] }
    private(set) var isLoaded = true

    func id(of kind: MuJoCoObjectKind, named name: String) -> Int? { nil }

    func bodyVelocity(bodyId: Int) -> [Double] { Array(repeating: 0, count: 6) }

    func load(sceneXML url: URL) throws { isLoaded = true }

    func reset() throws {
        time = 0
        qpos = Array(repeating: 0, count: sizes.nq)
        qvel = Array(repeating: 0, count: sizes.nv)
        control = Array(repeating: 0, count: sizes.nu)
    }

    func setState(qpos: [Double], qvel: [Double]) throws {
        self.qpos = qpos
        self.qvel = qvel
    }

    func write(controls: MotorControls) throws {
        control = controls.clipped().values
    }

    func stepPhysicsStep() throws {
        time += ModelStep.physicsTimestepSeconds
        let mean = control.reduce(0, +) / Double(max(control.count, 1))
        qpos[0] += 0.001 * mean
    }
}

/// A reader whose reading is 4 numbers: the simulated time, qpos[0], and their sine and
/// cosine.
final class PreviewFakeSenseReader: SenseReader {
    let sense: SenseName

    init(sense: SenseName) {
        self.sense = sense
    }

    func read(_ physics: any PhysicsReading, at step: ModelStep) throws -> SenseReading {
        let t = physics.time
        let q = physics.qpos.first ?? 0
        let primary = try ShapedArray(shape: [4], values: [Float(t), Float(q), Float(sin(t)), Float(cos(t))])
        return SenseReading(sense: sense, primary: primary)
    }
}

/// A tokenizer that passes the reading's arrays through unchanged.
struct PreviewFakeTokenizer: Tokenizer {
    let sense: SenseName

    func tokenize(_ reading: SenseReading) throws -> EncoderInput {
        guard reading.sense == sense else {
            throw SessionStepError.readingOfWrongSense(expected: sense, found: reading.sense)
        }
        return EncoderInput(sense: reading.sense, primary: reading.primary, auxiliary: reading.auxiliary, isFresh: reading.isFresh)
    }
}

/// A forward model whose token for sense k at model step n is sqrt(2) sin(0.1 i (k + 1) +
/// 0.3 n), root mean square near 1.0; its prediction is the same formula at n + 1 times
/// 0.9; its command is all zeros with strength 0.5; its wellness is cos(0.3 n).
final class PreviewFakeForwardModel: ForwardModeling {
    let contextLength = ModelStep.contextLength
    private(set) var filledContextSteps = 0

    func step(inputs: [SenseName: EncoderInput], previousExecutedCommand: Command?, at step: ModelStep) throws -> ModelOutput {
        var tokens: TokenSet = [:]
        var predicted: TokenSet = [:]
        for (k, sense) in SenseName.active.enumerated() {
            tokens[sense] = try PreviewFakeForwardModel.token(sense: k, at: Double(step.count))
            let next = try PreviewFakeForwardModel.token(sense: k, at: Double(step.count + 1))
            predicted[sense] = try Token(values: next.values.map { $0 * 0.9 })
        }
        var command = Command.zero(scale: .normalized)
        for handle in BodyHandle.allCases {
            command[handle, .quaternionW] = 1
            command[handle, .strength] = 0.5
        }
        filledContextSteps = min(filledContextSteps + 1, contextLength)
        return ModelOutput(
            step: step, tokens: tokens, predictedTokens: predicted, command: command,
            wellness: Float(cos(0.3 * Double(step.count))))
    }

    func reset() {
        filledContextSteps = 0
    }

    private static func token(sense k: Int, at n: Double) throws -> Token {
        let values = (0 ..< Token.count).map { i in
            Float(2.0.squareRoot() * sin(0.1 * Double(i) * Double(k + 1) + 0.3 * n))
        }
        return try Token(values: values)
    }
}

/// A planner whose goal ring keeps the last 200 (model step, wellness) pairs and whose
/// event returns the command it was given with 21 scores falling from 1.0 by 0.02 per
/// update, the last one selected, and the ring's highest-wellness entry as the goal.
final class PreviewFakePlanner: Planning {
    private var ring: [(step: ModelStep, wellness: Float)] = []

    var goalRingCount: Int { ring.count }

    func record(tokens: TokenSet, wellness: Float, at step: ModelStep) {
        ring.append((step: step, wellness: wellness))
        if ring.count > ModelStep.goalRingLength {
            ring.removeFirst(ring.count - ModelStep.goalRingLength)
        }
    }

    func plan(from command: Command, at step: ModelStep) throws -> PlanResult {
        let scores = (0 ..< PlanResult.evaluationsPerEvent).map { Float(1.0 - 0.02 * Double($0)) }
        let goal = ring.max { $0.wellness < $1.wellness }
        return try PlanResult(
            command: command,
            scores: scores,
            selectedIteration: scores.count - 1,
            goalStep: goal?.step ?? step,
            goalWellness: goal?.wellness ?? 0,
            fingerGoals: nil)
    }

    func reset() {
        ring.removeAll()
    }
}

/// A decoder whose control i is 0.5 sin(0.3 i + strength of wrist_L), within -1 to 1.
final class PreviewFakeDecoder: CommandDecoding {
    let motorContract = "ancestor_arm_axes_signed_limits_v3+finger_joint_servo_v1"

    func decode(_ command: Command, mask: HandleMask, gaze: GazeAngles?, fingerGoals: FingerJointGoals?) throws -> MotorControls {
        guard command.scale == .centimetres else {
            throw UEMACoreError.wrongScale(expected: .centimetres, found: command.scale)
        }
        let offset = Double(command[.wristL, .strength])
        let values = (0 ..< MotorControls.count).map { 0.5 * sin(0.3 * Double($0) + offset) }
        return try MotorControls(values: values)
    }
}

/// A stand-in for the execution transform: positions times 200 (cm), the other columns
/// copied. The released transform is not a multiplication (Command.swift:202-206); this
/// fake only gives the decoder a command at `.centimetres` scale.
final class PreviewFakeExecutor: CommandExecuting {
    func executedCommand(fromRaw raw: Command, achievedMotor: SenseReading, at step: ModelStep) throws -> Command {
        var values = raw.values
        for handle in BodyHandle.allCases {
            for column in CommandColumn.position {
                values[Command.index(handle: handle, column: column)] *= Command.positionRangeCentimetres
            }
        }
        return try Command(values: values, scale: .centimetres)
    }

    func reset() {}
}

/// Eyes that draw a diagonal colour pattern shifted by 8 pixels per render, so the five
/// images change every model step; no geoms.
final class PreviewFakeEyes: EyeRendering, SceneGeomReading {
    private var renders = 0

    func sceneGeoms() throws -> [SceneGeom] { [] }

    func cameraPose(_ camera: EyeCamera) throws -> CameraPose {
        try CameraPose(position: [0, 0, 0], rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1])
    }

    func render(camera: EyeCamera, pose: CameraPose, geoms: [SceneGeom]) throws -> EyeImage {
        if camera == .eyeLeft {
            renders += 1
        }
        let size = camera.pixelSize
        let shift = renders * 8
        let blue = UInt8((EyeCamera.allCases.firstIndex(of: camera) ?? 0) * 50)
        var rgb = [UInt8](repeating: 0, count: size * size * 3)
        for row in 0 ..< size {
            for column in 0 ..< size {
                let base = (row * size + column) * 3
                rgb[base] = UInt8((column + shift) % 256)
                rgb[base + 1] = UInt8(row % 256)
                rgb[base + 2] = blue
            }
        }
        return try EyeImage(camera: camera, width: size, height: size, rgb: rgb)
    }
}

/// The preview of this work item's acceptance line: the fake stack, one model step run when
/// the preview appears.
@MainActor
struct FakeStackOneModelStepPreview: View {
    @StateObject private var app = MindAppModel(
        builder: FakeStackSessionBuilder(),
        memory: MemoryViewModel(monitor: ProcInfoMemoryMonitor()))

    var body: some View {
        MindRootView(app: app)
            .onAppear {
                guard app.run == nil else { return }
                app.makeSession(weightsURL: nil)
                app.run?.stepOnce()
            }
    }
}

#Preview("One model step on the fake stack") {
    FakeStackOneModelStepPreview()
}
#endif
